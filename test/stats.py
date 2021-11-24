import numpy as np
import cv2
import time

A = [[1 , 2, 4, 1, 9],
[2 , 1, 4 , 1, 10]]

# 读入图片
img = cv2.imread("9318.jpg")
# 中值滤波，去噪
img = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 阈值分割得到二值化图片
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 膨胀操作
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_clo = cv2.dilate(binary, kernel2, iterations=2)
# 连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
contours, hierarchy = cv2.findContours(bin_clo,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
bin_clo = cv2.drawContours(bin_clo,contours,0,(10,100,100),-1)  
cv2.imwrite("9318+.jpg" , bin_clo)
# 查看各个返回值
# 连通域数量
print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
print('stats = ',stats)
# 连通域的中心点
# print('centroids = ',centroids)
# # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
# print('labels = ',labels.max())
print('连通域个数' , len(contours))
print(contours[1])
AA = np.vstack((contours[1],contours[0]))
print('========================')
print(AA)
print('========================')
print("AA shape:" , AA.shape)
C = np.array(contours[0]).reshape(-1,2)   
# print(C)
print("shape:" , C.shape)
print( 'first column [max , min] :' , np.max(C[:,0]) , np.min(C[:,0]))
print( 'first column [max , min] position :' , np.argmax(C[:,0] ) , np.argmin(C[:,0] ) )
print( 'point ( x max ) :' , C[np.argmax(C[:,0] ) , :])
print( 'point ( x min ) :' , C[np.argmin(C[:,0] ) , :])
print( 'second column [max , min] :' , np.max(C[:,1]) , np.min(C[:,1]))
print( 'second column [max , min] position:' , np.argmax(C[:,1]) , np.argmin(C[:,1]) )
print( 'point ( y max ) :' , C[np.argmax(C[:,1] ) , :])
print( 'point ( y min ) :' , C[np.argmin(C[:,1] ) , :])
print('time:' , str(time.time()).replace('.' , ''))
# D = []
# D.append(C[np.argmin(C[:,0] ) , :])
# D.append(C[np.argmax(C[:,0] ) , :])
# D.append(C[np.argmin(C[:,1] ) , :])
# D.append(C[np.argmax(C[:,1] ) , :])
D = np.array( [ C[np.argmin(C[:,0] ) , :] , C[np.argmax(C[:,0] ) , :] , C[np.argmin(C[:,1] ) , :] , C[np.argmax(C[:,1] ) , :] ] )
print('D:' , D.shape , D)
print(D[2,1])
