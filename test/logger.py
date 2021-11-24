'''
Description:  测试logger和list的一些操作
FilePath: /BeiLu_PostProcess/testfile/logger.py
Author: Rainche
Date: 2021-09-07 11:11:45
LastEditTime: 2021-11-15 11:07:49
'''
import numpy as np
list1 = [1,2,3]
list2 = [1,2,3,4]
list3 = []
# print([9 for i in range(len(list1))])

# list3 = list3 + list2
# print(list3)

A = -1 * np.ones((len(list1) , len(list2)))

row_ones = 999 * np.ones((1 , len(list2)))
col_ones = 999 * np.ones((len(list1) + 1 , 1))
# print(row_ones)
# print(col_ones)
B = np.array([[-3, 5, 7,1],[12, 7, 4, 8],[3, -6, -1, 0]])
print(B)
B = np.vstack((B , row_ones))
B = np.hstack((B, col_ones))
print(B)
# for i in range(len(list1)):
#     for j in range(len(list2)):
#         print(B[i][j])
min_index = np.unravel_index(np.argmin(B, axis=None), B.shape)
min_value = B.min()
print(min_index, min_value)
pairs = []
pairs.append(min_index)
pairs.append(min_index)
# print(B[pairs[0]])
# B[min_index[0]] = B[-1]
# B[:,min_index[1]] = B[:,-1]
# print(B)
for each_pair in pairs:
    print(B[each_pair])

# import sys
# import os
# class Logger(object):
#   def __init__(self, filename="Default.log"):
#     self.terminal = sys.stdout
#     self.log = open(filename, "a")
#   def write(self, message):
#     # self.terminal.write(message)
#     self.log.write(message)
#   def flush(self):
#     pass
# # path = os.path.abspath(os.path.dirname(__file__))
# # type = sys.getfilesystemencoding()
# sys.stdout = Logger('a.txt')
# # print(path)
# # print(os.path.dirname(__file__))
# print('------------')
