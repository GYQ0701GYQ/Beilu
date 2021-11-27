'''
Description: 
FilePath: /beilu/test/deque_test.py
Autor: Rainche
Date: 2021-11-23 16:17:44
LastEditTime: 2021-11-25 21:44:24
'''
# from collections import deque
# A = deque(maxlen=10)
# for i in range(5) :
#     A.append(i)
# A[3] = 100
# print(A)
# for i in range(0, 2) :
#     A[i] = A[0]
# print(A)
import numpy as np
seq1 = np.arange(5)
print(seq1)
seq2 = []
print(np.concatenate((seq1,seq2)).astype(np.uint16))
