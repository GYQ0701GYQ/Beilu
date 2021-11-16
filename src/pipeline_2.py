import ctypes
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_path = '/data/guanyuqi/beilu'
# for Yolov5
sys.path.append(os.path.join(base_path, 'src/yolov5'))
# for yolact_edge
sys.path.append(os.path.join(base_path, 'src/yolact_edge'))

# for yolact_edge
from yolact_edge.data import COCODetection, YoutubeVIS, get_label_map, MEANS, COLORS
from yolact_edge.data import cfg, set_cfg, set_dataset
from yolact_edge.yolact import Yolact
from yolact_edge.utils.augmentations import BaseTransform, BaseTransformVideo, FastBaseTransform, Resize
from yolact_edge.utils.functions import MovingAverage, ProgressBar
from yolact_edge.layers.box_utils import jaccard, center_size
from yolact_edge.utils import timer
from yolact_edge.utils.functions import SavePath
from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.utils.tensorrt import convert_to_tensorrt
from eval import str2bool, parse_args, Detections, prep_coco_cats
# for yolov5
from yolov5_trt_beilu2 import get_img_path_batches, plot_one_box, YoLov5TRT, inferThread

import pycocotools

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import threading
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import matplotlib.pyplot as plt
import cv2
import logging
import math
from tqdm import tqdm

class Pipeline():
    def __init__(self):
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.yolo_init()
        self.yolact_edge_init()
        
        

    def yolact_edge_init(self):
        self.args = parse_args()
        self.args.trained_model = '/data/guanyuqi/beilu/src/yolact_edge/models/yolact_edge_beilu_mobilenetv2_1351_50000.pth' #/yolact_beilu_540_20000.pth'
        # self.args.use_tensorrt_safe_mode = True
        self.args.disable_tensorrt = True
        self.args.benchmark = True
        self.args.config = 'yolact_edge_beilu_mobilenetv2_config' #'yolact_edge_beilu_config'
        set_cfg(self.args.config)

        ### added:
        # if os.path.exists(self.args.trained_model):
        #     dir_path = '/'.join(self.args.trained_model.split('/')[:-1])
        #     file_name = self.args.trained_model.split('/')[-1]
        #     for f in os.listdir(dir_path):
        #         if len(f)>len(file_name) and f[:len(file_name)] == file_name and f[len(file_name)] == '.' and f[-3:] == 'trt':
        #             os.remove(os.path.join(dir_path,f))

        from yolact_edge.utils.logging_helper import setup_logger
        setup_logger(logging_level=logging.INFO)
        logger = logging.getLogger("yolact.eval")

        with torch.no_grad():
            # if self.args.cuda:
            # cudnn.benchmark = True
            # cudnn.fastest = True
                # if self.args.deterministic: # False
                #     cudnn.deterministic = True
                #     cudnn.benchmark = False
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # else:
            #     torch.set_default_tensor_type('torch.FloatTensor')

            logger.info('Loading model...')
            self.yolact = Yolact(training=False)
            self.yolact.load_weights(self.args.trained_model, args=self.args)
            self.yolact.eval()
            logger.info('Model loaded.')

            convert_to_tensorrt(self.yolact, cfg, self.args, transform=BaseTransform())
            if self.args.cuda:
                self.yolact = self.yolact.cuda()

            self.yolact.detect.use_fast_nms = self.args.fast_nms
            cfg.mask_proto_debug = self.args.mask_proto_debug

        return

    def yolact_detect(self, img):
        h, w, _ = img.shape
        # if args.output_coco_json: # TODO
        #     detections.dump()
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0)) # 96.5

        # extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None} #108
        extras = {"backbone": "full", "interrupt": False, "moving_statistics": {"aligned_feats": []}}
        preds = self.yolact(batch, extras=extras)["pred_outs"]
        
        
        t = postprocess(preds, w, h, crop_masks = self.args.crop, score_threshold = self.args.score_threshold)
        top_k = 1
        classes, scores, boxes, masks = [x[:top_k].detach().cpu().numpy() for x in t]
        # classes, scores, boxes, masks = [x[:top_k].cpu().numpy() for x in t]
        torch.cuda.synchronize()
        return classes, scores, boxes, masks
        # img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

        # if args.output_coco_json:
        #     with timer.env('Postprocess'):
        #         _, _, h, w = batch.size()
        #         classes, scores, boxes, masks = \
        #             postprocess(preds, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

        #     with timer.env('JSON Output'):
        #         boxes = boxes.cpu().numpy()
        #         masks = masks.view(-1, h, w).cpu().numpy()
        #         for i in range(masks.shape[0]):
        #             # Make sure that the bounding box actually makes sense and a mask was produced
        #             if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
        #                 detections.add_bbox(image_id, classes[i], boxes[i,:],   scores[i])
        #                 detections.add_mask(image_id, classes[i], masks[i,:,:], scores[i])

        # if save_path is None:
        #     img_numpy = img_numpy[:, :, (2, 1, 0)]
        #     plt.imshow(img_numpy)
        #     plt.title(path)
        #     plt.show()
        # else:
        #     cv2.imwrite(save_path, img_numpy)

    def yolo_init(self):
        # self.yolo_conf_thresh = 0.5
        # self.yolo_iou_thresh = 0.4
        self.yolo_plugin_library = os.path.join(base_path, 'src/yolov5', "build/libmyplugins.so")
        self.yolo_engine_file_path = os.path.join(base_path, 'src/yolov5', "build/yolov5s_beilu.engine")
        ctypes.CDLL(self.yolo_plugin_library)
        # self.yolo_categories = ["person"]
        self.yolov5_wrapper = YoLov5TRT(self.yolo_engine_file_path)
        print('batch size is {}, yolov5 initialized.'.format(self.yolov5_wrapper.batch_size))

    def yolo_detect(self, img):
        # thread1 = inferThread(self.yolov5_wrapper, img)
        # thread1.start()
        # thread1.join()
        # yolo_boxes, yolo_scores, use_time = thread1.result()
        yolo_boxes, yolo_scores, _ = self.yolov5_wrapper.infer(img)
        return yolo_boxes, yolo_scores

    def detect(self, img):
        _, yolact_scores, yolact_boxes, yolact_masks = self.yolact_detect(img) # boxes:l,t,r,b
        if yolact_masks is not None and len(yolact_masks)>0:
            yolact_mask = yolact_masks[0]
        else:
            yolact_mask = None
        yolo_boxes, yolo_scores = self.yolo_detect(img) # boxes:l,t,r,b
        if not len(yolo_boxes)>0:
            yolo_boxes = None
        return yolact_mask, yolo_boxes
    
    # def single_img_save(self, img, masks, boxes, file_name):
    #     # TODO: add masks
    #     np.savetxt(os.path.join('person_box_result',file_name+'.txt'), boxes, fmt="%d")
    #     return
    
    def single_img_visual(self, img, mask, boxes, show=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if boxes is not None:
            for box in boxes:
                cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])),(0, 255, 0), 3)
        if mask is not None:
            mask = mask.astype(np.uint8)
            mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
            img = cv2.addWeighted(img, 0.77, mask, 0.23, -1)
        if show:
            cv2.imshow('pipeline', img)
            cv2.resizeWindow('pipeline', 1080, 720)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

def inflation_corrosion(mask , threshold_point , area ):
    kernel = np.ones(area , np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def contours_approx(mask , img , min_epsilon):
    approx = []
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    if len(contours) <= 0:
        result_img = cv2.putText(img, 'safe' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return result_img , approx
    max_len = 0
    cnt = []
    for i in range(len(contours)):
        if max_len < len(contours[i]):
            max_len = len(contours[i])
            cnt = contours[i]
    if len(cnt) <= 0:
        result_img = cv2.putText(img, 'safe' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return result_img , approx
    else :
        approx1 = cv2.approxPolyDP(cnt, min_epsilon, True)
        return img , approx1

def handle_approx_line(approx , img , boxes ):
    [vx, vy, x, y] = cv2.fitLine(approx, cv2.DIST_L2, 0, 0.01, 0.01)
    rows, cols = img.shape[:2]
    if vx == 0 :
        vx += 0.0001
    if vy == 0 :
        vy += 0.0001
    k = vy/vx 
    b = y - k * x
    left_y = int((-x*vy/vx) + y)
    right_y = int(((cols-x)*vy/vx) + y)
    top_x = int(x - y*vx/vy)
    bottom_x = int(x + (rows - y)*vx/vy)
    cv2.line(img, (top_x, 0), (bottom_x, rows - 1), (0, 255, 0), 4)

    right = 0
    for each_point in approx :
        if each_point[0][0] > math.floor(cols / 2) :
            right += 1
    if right > math.floor(len(approx) / 2) :
        flag = "right"
    else :
        flag = 'left'
    if left_y > 0 and left_y < rows and right_y > 0 and right_y < rows :
        flag = "below"
    elif k < 0 and top_x > math.floor(cols / 2) and top_x < cols and bottom_x > 0 and bottom_x < cols :
        flag = "right"
    elif k < 0 and bottom_x > 0 and bottom_x < cols and right_y > 0 and right_y < rows :
        flag = "right"
    elif k > 0 and top_x > 0 and top_x < math.floor(cols / 2) and bottom_x > 0 and bottom_x < cols :
        flag = "left"
    elif k > 0 and bottom_x > 0 and bottom_x < cols and left_y > 0 and left_y < rows :
        flag = "left"

    warning_tag = 'safe'
    if boxes is not None :
        for box in boxes :
            piont2 = (int((box[0]+box[2]) / 2 ) , box[3] )
            judeg_pos = {'left': 1 , 'right': -1}
            if flag == "below" :
                if k * piont2[0] + b > piont2[1] :
                    warning_tag = 'warning'
                    break
            elif flag == 'left' or flag == 'right' :
                if judeg_pos[flag]*(( piont2[1] - b ) / k - piont2[0]) < 0 :
                    warning_tag = 'warning'
                    break
    if warning_tag == 'safe' :
        result_img = cv2.putText(img, 'safe' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    else :
        result_img = cv2.putText(img, 'warning' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    if flag is not None:
        result_img = cv2.putText(result_img, flag , (10,150 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return result_img

def post_process(img , mask, boxes):
    if mask is None :
        result_img = cv2.putText(img, 'safe' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return result_img
    if boxes is not None :
        boxes = boxes.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255
    mask = inflation_corrosion(mask , 10 , (10 , 10))        
    img , approx_res = contours_approx(mask , img , 5)
    if len(approx_res) <= 0 :
        return img
    else :
        result_img = handle_approx_line(approx_res , img , boxes )
        return result_img
    

if __name__=='__main__':
    pipeline = Pipeline()

    # # json_file_path = '/nas/datasets/beilu/beilu_anno_val.json'
    # json_file_path = '/nas/datasets/beilu/beilu_person_anno_val.json'
    # with open(json_file_path, 'r') as f:
    #     json_data = json.load(f)
    # imgs = json_data['images']
    # # img_base_path = '/nas/datasets/beilu/labeled_data'
    # img_base_path = '/nas/datasets/beilu/labeled_data/person/JPEGImages'
    # for img in tqdm(imgs):
    #     path = os.path.join(img_base_path, img['file_name'])
    #     # out_path = os.path.join(base_path, output_folder, img['file_name'])
    #     img = cv2.imread(path)
    #     ctrnet_boxes = pipeline.detect(img)
    #     pipeline.single_img_save(img, None, ctrnet_boxes, img['file_name'])
    video_path = '/nas/datasets/beilu/2.视频识别（17个录像）/010-20201119-付村-清晰-人员经过支架.mp4'
    video_out_path = '/data/guanyuqi/beilu/out_video_010.mp4'
    print("Reading video file...")
    cam = cv2.VideoCapture(video_path)
    target_fps   = round(cam.get(cv2.CAP_PROP_FPS))
    frame_width  = round(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # 2560
    frame_height = round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1440
    num_frames   = round(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
    frame_counter = 0
    timer = 0
    print("Detecting...")
    test_flag = True
    while True:
        _, img = cam.read()
        if img is None:
            break
        start_time = time.time()
        yolact_mask, ctrnet_boxes = pipeline.detect(img)
        if frame_counter!=0:
            timer += time.time()-start_time
        frame_counter += 1
        print(frame_counter, end=" ")
        frame = pipeline.single_img_visual(img, yolact_mask, ctrnet_boxes)
        frame = post_process(frame, yolact_mask, ctrnet_boxes)
        out.write(frame)
    print('Done.')
    print("Total time: {}; frames: {}-1; {} ms".format(timer, frame_counter, timer/(frame_counter-1)*1000))
    cam.release()
    out.release()
    pipeline.yolov5_wrapper.destroy()