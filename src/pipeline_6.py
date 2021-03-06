'''
Description:  
Author: Rainche
Date: 2021-11-07 11:31:16
LastEditTime: 2021-11-07 23:46:16
LastEditors: Rainche
'''
import sys
import os
import re
base_path = '/data/guanyuqi/beilu'
res_path = '/data/guanyuqi/beilu/result/pic_005_countors6/'
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
import math

class Pipeline():
    def __init__(self):
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.yolact_edge_init()
    
    def yolact_edge_init(self):
        self.args = parse_args()
        # self.args.trained_model = os.path.join(base_path, 'model','yolact_edge_beilu_all_mobilenetv2_382_80000.pth')
        self.args.trained_model = os.path.join('/nas/xuxiang/YOLACT_pretrained_model','yolact_beilu_all_384_40000.pth')
        # self.args.trained_model = os.path.join('/nas/xuxiang/YOLACT_pretrained_model','yolact_beilu_all_96_10000.pth')
        self.args.use_tensorrt_safe_mode = True
        # self.args.disable_tensorrt = True
        self.args.benchmark = True
        self.args.config = 'yolact_edge_beilu_all_config' # 'yolact_edge_beilu_all_mobilenetv2_config' #
        self.args.score_threshold = 0.15
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
        # ????????????
        dlc_masks, person_masks, person_boxes = [], [], []
        for i in range(len(classes)):
            if classes[i] == 0:
                dlc_masks.append(masks[i])
            elif classes[i] == 1:
                person_masks.append(masks[i])
                person_boxes.append(boxes[i])
        
        return dlc_masks, person_masks, person_boxes
    
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

def inflation_corrosion(mask , threshold_point , area ):
    kernel = np.ones(area , np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def contours_approx(mask , img , epsilon):
    if_write = False # flag for second contours
    
    approx = []
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    if len(contours) <= 0:
        result_img = cv2.putText(img, 'safe' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return result_img , approx , if_write
    elif len(contours) >= 2 :
        if_write = True

    max_len = 0
    second_len = 0
    cnt = []
    second_cnt = []
    for i in range(len(contours)):
        if max_len < len(contours[i]):
            max_len = len(contours[i])
            cnt = contours[i]
        elif max_len > len(contours[i])  and second_len < len(contours[i]) :
            second_len = len(contours[i])
            second_cnt = contours[i]
    
    if len(second_cnt) > 0 :   # handle second mask
        # img = cv2.drawContours(img,cnt,-1,(255,0,0),5)   # optional
        points_A = np.array(second_cnt).reshape(-1,2)   
        A = np.array( [ points_A[np.argmin(points_A[:,0] ) , :] , points_A[np.argmax(points_A[:,0] ) , :] , points_A[np.argmin(points_A[:,1] ) , :] , points_A[np.argmax(points_A[:,1] ) , :] ] )
        points_B = np.array(cnt).reshape(-1,2)   
        B = np.array( [ points_B[np.argmin(points_B[:,0] ) , :] , points_B[np.argmax(points_B[:,0] ) , :] , points_B[np.argmin(points_B[:,1] ) , :] , points_B[np.argmax(points_B[:,1] ) , :] ] )        
        if  ( A[0,0] >= (B[1,0] - 30 ) and A[2,1] >= (B[3,1]-30) and 2*(A[2,0] - B[3,0]) < (B[1,0] - B[0,0]) ) or\
         ( B[0,0] >= (A[1,0] - 30) and B[2,1] >= (A[3,1] - 30) and 2*(B[2,0] - A[3,0]) < (B[1,0] - B[0,0]) ) :
            # cnt = np.vstack((cnt,second_cnt))
            img = cv2.drawContours(img,second_cnt,-1,(255,0,0),3)   # optional
            # img = cv2.putText(img, '2 region' , (10,300 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)   # optional
        else :
            second_cnt = []


    if len(second_cnt) > 0:
        approx1 = cv2.approxPolyDP(cnt, epsilon, True)
        approx2 = cv2.approxPolyDP(second_cnt, 0.2, True)
        approx = np.vstack((approx1,approx2))
        img = cv2.putText(img, str(len(approx1)) + ':' + str(len(approx2)) + ':' + str(len(approx)) , (10,300 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return img , approx , if_write
    else :
        approx1 = cv2.approxPolyDP(cnt, epsilon, True)
        return img , approx1 , if_write

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
    if_write = False
    if mask is None :
        result_img = cv2.putText(img, 'safe' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return result_img , if_write
    # if boxes is not None :
    #     boxes = boxes.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255
    mask = inflation_corrosion(mask , 10 , (10 , 10))        
    img , approx_res , if_write = contours_approx(mask , img , 5)
    if len(approx_res) <= 0 :
        return img , if_write
    else :
        result_img = handle_approx_line(approx_res , img , boxes )
        return result_img , if_write

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

if __name__=='__main__':
    argv_list = []
    print('???????????????????????????????????????????????????????????????????????????,??????????????????????????????????????????')
    temp = input()
    argv_list = re.split(',' , temp)

    pipeline = Pipeline()
    frame_counter = 0
    timer = 0
    timer_post = 0

    if len(argv_list) == 1 and argv_list[0] == 'mov' :
        argv_list.append(r'/nas/datasets/beilu/2.???????????????17????????????/005-D1-131-??????-??????-?????????????????????-????????????2.mp4')
        argv_list.append(r'/data/guanyuqi/beilu/result/res_005_countors_6.mp4')
    if len(argv_list) >= 3 and argv_list[0] == 'mov' :
        video_path = argv_list[1]
        video_out_path = argv_list[2]
        print("Reading video file...")
        cam = cv2.VideoCapture(video_path)
        target_fps   = round(cam.get(cv2.CAP_PROP_FPS))
        frame_width  = round(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        frame_height = round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames   = round(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
    
        print("Detecting...")
        while True:
            _, img = cam.read()
            if img is None:
                break
            start_time = time.time()
            dlc_masks, _, person_boxes = pipeline.yolact_detect(img)
            if frame_counter!=0:
                timer += time.time()-start_time
            frame_counter += 1
            print(frame_counter, end=" ")
            if len(dlc_masks)>0:
                dlc_mask = dlc_masks[0]
            else:
                dlc_mask = None
            frame = pipeline.single_img_visual(img, dlc_mask, person_boxes)

            start_time_post = time.time()
            frame , if_write = post_process(frame, dlc_mask, person_boxes)
            timer_post += time.time()-start_time_post

            if if_write==True:
                temp_name = str(frame_counter) + '.jpg'
                # temp_name = str(time.time()).replace('.' , '') + '.jpg'
                pic_path = os.path.join(res_path, temp_name)
                cv2.imwrite(pic_path , frame)               

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
            img_result = img_path.replace(img_in_path , img_out_path)
            start_time = time.time()
            dlc_masks, _, person_boxes = pipeline.yolact_detect(img)
            if frame_counter!=0:
                timer += time.time()-start_time
            frame_counter += 1
            print(frame_counter, end=" ")
            if len(dlc_masks)>0:
                dlc_mask = dlc_masks[0]
            else:
                dlc_mask = None
            frame = pipeline.single_img_visual(img, dlc_mask, person_boxes)
            frame = post_process(frame, dlc_mask, person_boxes)
            cv2.imwrite(img_result , frame)
    print('Done.')
    if frame_counter > 1 :
        print("Detect time: {}; frames: {}-1; {} ms".format(timer, frame_counter, timer/(frame_counter-1)*1000))
        print("Post time: {}; frames: {}-1; {} ms".format(timer_post, frame_counter, timer_post/(frame_counter-1)*1000))
    