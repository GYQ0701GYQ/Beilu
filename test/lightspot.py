import cv2
import numpy as np
import math
import time


def distance(A , B):
    return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

white_unit = np.array([255,255,255] , dtype=np.uint8)
ONES = np.ones(3 , dtype=np.uint8)
center = [np.random.randint(0,1440) , np.random.randint(0,2560)]
print('center_point' , center)
radius = np.random.randint(low=30, high=120)
radius23 = math.floor(radius*2 / 3)
double_radius = radius * 2                                                                     
tri_radius = radius * 3
print('radius' , radius)

img = cv2.imread("ls1.jpg")
print('origin img:' , img.shape , img.dtype , img[150][397] ,  img[150,397] ,img[150][397].dtype)

time_start=time.time()

for row in range(max(0 , center[0] - 3 * radius) ,min( len(img) , center[0] + 3 * radius ) ) :
    for col in range(max(0 , center[1] - 3 * radius) ,min( len(img[0]) , center[1] + 3 * radius )) :
        dis = distance(center , [row , col])
        if dis < radius23 :
            # img.itemset((row,col,0),255)
            # img.itemset((row,col,1),255)
            # img.itemset((row,col,2),255)
            img[row,col] = white_unit
        elif dis < radius :
            radio = ( dis - radius23 ) / (radius - radius23)
            transit_unit = ( 255 - math.floor(55 * radio) ) * ONES
            img[row,col] = transit_unit
            # img.itemset((row,col,0), 255 - temp)
            # img.itemset((row,col,1), 255 - temp)
            # img.itemset((row,col,2), 255 - temp)
        # elif dis < radius :
        #     img.itemset((row,col,0),255)
        #     img.itemset((row,col,1),255)
        #     img.itemset((row,col,2),255)
        # elif dis >= radius and dis <= double_radius :
        #     radio = ( dis - radius ) / radius
        #     temp = math.floor(55 * radio)
        #     img.itemset((row,col,0), 255 - temp)
        #     img.itemset((row,col,1), 255 - temp)
        #     img.itemset((row,col,2), 255 - temp)
        # elif dis <= tri_radius :
        #     radio = ( dis - double_radius ) / radius
        #     img.itemset((row,col,0), math.floor(radio*img.item(row,col,0) + (1-radio)*200 ))
        #     img.itemset((row,col,1), math.floor(radio*img.item(row,col,1) + (1-radio)*200 ))
        #     img.itemset((row,col,2), math.floor(radio*img.item(row,col,2) + (1-radio)*200 ))
        elif dis <= tri_radius :
            radio = (dis - radius) / (2 * radius)
            # gradient_unit = math.floor( radio*img[row,col] + (1-radio)*200*ONES)
            gradient_unit = (radio*img[row,col] + (1-radio)*200*ONES).astype(np.uint8)
            # print(gradient_unit.dtype)
            img[row,col] = gradient_unit
            # img.itemset((row,col,0), math.floor(radio*img.item(row,col,0) + (1-radio)*200 ))
            # img.itemset((row,col,1), math.floor(radio*img.item(row,col,1) + (1-radio)*200 ))
            # img.itemset((row,col,2), math.floor(radio*img.item(row,col,2) + (1-radio)*200 ))

time_end=time.time()
print('time cost',time_end-time_start,'s')
# print(img.dtype)
cv2.imwrite('ls1_plus.jpg' , img)


