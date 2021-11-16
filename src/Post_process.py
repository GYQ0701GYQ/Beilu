'''
Description: 
FilePath: /beilu/src/Post_process.py
Autor: Rainche
Date: 2021-11-16 17:25:25
LastEditTime: 2021-11-16 22:07:29
'''
import cv2
import numpy as np
import math

def inflation_corrosion(mask , threshold_point , area ):
    kernel = np.ones(area , np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def contours_approx(mask , img , epsilon):
    approx = []
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    if len(contours) <= 0:
        result_img = cv2.putText(img, 'safe' , (10,200 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        return result_img , approx
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
        points_A = np.array(second_cnt).reshape(-1,2)   
        A = np.array( [ points_A[np.argmin(points_A[:,0] ) , :] , points_A[np.argmax(points_A[:,0] ) , :] , points_A[np.argmin(points_A[:,1] ) , :] , points_A[np.argmax(points_A[:,1] ) , :] ] )
        points_B = np.array(cnt).reshape(-1,2)   
        B = np.array( [ points_B[np.argmin(points_B[:,0] ) , :] , points_B[np.argmax(points_B[:,0] ) , :] , points_B[np.argmin(points_B[:,1] ) , :] , points_B[np.argmax(points_B[:,1] ) , :] ] )        
        if  ( A[0,0] >= (B[1,0] - 30 ) and A[2,1] >= (B[3,1]-30) and 2*(A[2,0] - B[3,0]) < (B[1,0] - B[0,0]) ) or\
         ( B[0,0] >= (A[1,0] - 30) and B[2,1] >= (A[3,1] - 30) and 2*(B[2,0] - A[3,0]) < (B[1,0] - B[0,0]) ) :
            # img = cv2.drawContours(img,second_cnt,-1,(255,0,0),3)   # optional
            choose = 1
        else :
            second_cnt = []

    if len(second_cnt) > 0:
        approx1 = cv2.approxPolyDP(cnt, epsilon, True)
        approx2 = cv2.approxPolyDP(second_cnt, 0.2, True)
        approx = np.vstack((approx1,approx2))
        return img , approx
    else :
        approx1 = cv2.approxPolyDP(cnt, epsilon, True)
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
    # cv2.line(img, (top_x, int(0)), (bottom_x, int(rows - 1)), (0, 255, 0), 4)

    vertex = []
    for x in [[top_x , 0] , [ bottom_x , rows]] :
        if x[0] >= 0 and x[0] <= cols :
            vertex.append(x)
    for y in [[0 , left_y] , [cols , right_y]]:
        if y[1] >= 0 and y[1] <= rows:
            vertex.append(y)
    if len(vertex) >= 2 :
        cv2.line(img, vertex[0], vertex[1], (0, 255, 0), 4)
    Vertex = np.array(vertex)

    midpoint = (Vertex.mean(axis=0)).astype(np.int16)
    if midpoint[0] >= math.floor(cols / 2) and  midpoint[0] <= cols :
        flag = "right"
    elif midpoint[0] < math.floor(cols / 2) and  midpoint[0] >= 0 :
        flag = "left"
    if left_y > 0 and left_y < rows and right_y > 0 and right_y < rows :
        flag = "below"
    elif k < 0 and top_x > math.floor(2*cols / 3) and top_x < cols and bottom_x > 0 and bottom_x < cols :
        flag = "right"
    elif k < 0 and bottom_x > 0 and bottom_x < cols and right_y > 0 and right_y < rows :
        flag = "right"
    elif k > 0 and top_x > 0 and top_x < math.floor(cols / 3) and bottom_x > 0 and bottom_x < cols :
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
    # if boxes is not None :
    #     boxes = boxes.numpy()
    mask = mask.astype(np.uint8)
    mask *= 255
    mask = inflation_corrosion(mask , 10 , (10 , 10))        
    img , approx_res = contours_approx(mask , img , 5)
    if len(approx_res) <= 0 :
        return img
    else :
        result_img = handle_approx_line(approx_res , img , boxes )
        return result_img