import cv2
import numpy as np
import math
import time

img = cv2.imread("ls1.jpg")
# print('origin img:' , img.shape , img.dtype , img[150][397] ,  img[150,397] ,img[150][397].dtype)

time_start=time.time()

def illum(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , mask = cv2.threshold(img_bw, 225, 255, 0)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8)
    # img[mask == 255] = 150
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img_zero[y:y+h, x:x+w] = 255
    mask = img_zero
    result = cv2.illuminationChange(img, mask, alpha=1, beta=2)
    return result

result = illum(img)

time_end=time.time()
print('time cost',time_end-time_start,'s')
# print(img.dtype)
cv2.imwrite('ls1_minus.jpg' , result)


