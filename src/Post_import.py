'''
Description: 
FilePath: /beilu/src/Post_import.py
Autor: Rainche
Date: 2021-11-16 17:26:07
LastEditTime: 2021-11-16 22:08:20
'''
from Post_Process import post_process

import sys
import os
import re
base_path = '/data/guanyuqi/beilu'
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

import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import logging

def histogram(image):
    output_img = image.copy()
    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [256], [0,256])
        channel_result = cv2.equalizeHist(image[:,:,i])
        output_img[:,:,i] = channel_result
    return output_img.astype(np.float32)

class Pipeline():
    def __init__(self):
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.yolact_edge_init()
    
    def yolact_edge_init(self):
        self.args = parse_args()
        self.args.trained_model = os.path.join('/nas/xuxiang/YOLACT_pretrained_model',  'beilu_all_resnet50_randomspotlight3/yolact_edge_beilu_all_resnet50_851_80000.pth') 
        # 'beilu_all_resnet50/yolact_edge_beilu_all_resnet50_851_80000.pth'
        # 'beilu_all_resnet50/yolact_edge_beilu_all_resnet50_239_22500.pth'
        # 'beilu_all_resnet50_randomspotlight3/yolact_edge_beilu_all_resnet50_239_22500.pth'
        # 'beilu_all_resnet50_randomspotlight3/yolact_edge_beilu_all_resnet50_851_80000.pth'
        self.args.use_tensorrt_safe_mode = True
        # self.args.disable_tensorrt = True
        self.args.benchmark = True
        self.args.config = 'yolact_edge_beilu_all_resnet50_config' # 'yolact_edge_beilu_all_mobilenetv2_config' #'yolact_edge_beilu_all_config'
        self.args.score_threshold = 0.2
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
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))

        # extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None} #108
        extras = {"backbone": "full", "interrupt": False, "moving_statistics": {"aligned_feats": []}}
        preds = self.yolact(batch, extras=extras)["pred_outs"]
        
        t = postprocess(preds, w, h, crop_masks = self.args.crop, score_threshold = self.args.score_threshold)
        top_k = 8
        classes, scores, boxes, masks = [x[:top_k].detach().cpu().numpy() for x in t]
        torch.cuda.synchronize()
        # print(classes)
        # 分类处理
        dlc_masks, dlc_boxes, person_masks, person_boxes = [], [], [], []
        for i in range(len(classes)):
            if classes[i] == 0:
                dlc_masks.append(masks[i])
                dlc_boxes.append(boxes[i])
            elif classes[i] == 1:
                person_masks.append(masks[i])
                person_boxes.append(boxes[i])
        
        return dlc_masks, dlc_boxes, person_masks, person_boxes
    
    def single_img_visual(self, img, mask, boxes, show=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(boxes) > 0:
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

def get_file_path(root_path):
    file_list = []
    dir_list = []
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)            
            temp_file_list , temp_dir_list = get_file_path(dir_file_path)
            file_list += temp_file_list
            dir_list += temp_dir_list
        else:
            file_list.append(dir_file_path)
    return file_list,dir_list   

def main(pipeline , img , timer ,  timer_post , frame_counter):
    start_time = time.time()
    input_img = img.copy()
    # input_img = histogram(img)
    dlc_masks, _, _, person_boxes = pipeline.yolact_detect(input_img)
    if frame_counter!=0:
        timer += time.time()-start_time
    frame_counter += 1
    print(frame_counter, end=" ")

    if len(dlc_masks)>0:
        # simple version
        # dlc_mask = dlc_masks[0]
        # new version
        dlc_mask = np.zeros(dlc_masks[0].shape).astype(np.uint8)
        for m in range(len(dlc_masks)):
            dlc_mask = dlc_mask|dlc_masks[m].astype(np.uint8)
        # _, _, stats, _ = cv2.connectedComponentsWithStats(dlc_mask)
        # stats = stats[stats[:,4].argsort()]
        # dlc_box = [stats[-2,0], stats[-2,1], stats[-2,0]+stats[-2,2], stats[-2,1]+stats[-2,3]]
    else:
        dlc_mask, dlc_box = None, None

    frame = pipeline.single_img_visual(img, dlc_mask, person_boxes)

    start_time_post = time.time()
    frame = post_process(frame, dlc_mask, person_boxes)
    timer_post += time.time()-start_time_post
    return frame , timer ,  timer_post ,  frame_counter

if __name__=='__main__':
    argv_list = []
    print('运行模式，输入文件路径，输出文件路径：（用逗号隔开,文件或路径名中不可含有逗号）')
    temp = input()
    argv_list = re.split(',' , temp)

    pipeline = Pipeline()
    frame_counter = 0
    timer = 0
    timer_post = 0

    if len(argv_list) == 1 and argv_list[0] == 'mov' :
        argv_list.append(r'/nas/datasets/beilu/2.视频识别（17个录像）/010-20201119-付村-清晰-人员经过支架.mp4')
        argv_list.append(r'/data/guanyuqi/beilu/result/out_010_randomspotlight3_postmdf_851.mp4')
    if len(argv_list) >= 3 and argv_list[0] == 'mov' :
        video_path = argv_list[1]
        video_out_path = argv_list[2]
        print("Reading video file...")
        cam = cv2.VideoCapture(video_path)
        target_fps   = round(cam.get(cv2.CAP_PROP_FPS))
        frame_width  = round(cam.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        frame_height = round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
        num_frames   = round(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
        print("Detecting...")
        while True:
            _, img = cam.read()
            if img is None:
                break
            frame , timer ,  timer_post , frame_counter = main(pipeline , img , timer ,  timer_post , frame_counter)
            frame = cv2.resize(frame, (frame_width, frame_height))
            out.write(frame)
        cam.release()
        out.release()
    elif len(argv_list) >= 3 :
        imgs_list = []
        img_in_path = argv_list[1]
        img_out_path = argv_list[2]
        if argv_list[0] == 'img' :
            imgs_list.append(img_in_path)
        elif argv_list[0] == 'imgs' :
            imgs_list,_ = get_file_path(img_in_path)
        for img_path in imgs_list :
            img = cv2.imread(img_path)
            img_result_path = img_path.replace(img_in_path , img_out_path)
            if img is None:
                break
            frame , timer ,  timer_post , frame_counter = main(pipeline , img , timer ,  timer_post , frame_counter)
            cv2.imwrite(img_result_path , frame)
    print('Done.')
    if frame_counter > 1 :
        print("Total time: {}; frames: {}-1; {} ms".format(timer, frame_counter, timer/(frame_counter-1)*1000))
        print("Post time: {}; frames: {}-1; {} ms".format(timer_post, frame_counter, timer_post/(frame_counter-1)*1000))

