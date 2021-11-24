'''
Description: 
FilePath: /beilu/test/post_test.py
Autor: Rainche
Date: 2021-11-16 20:49:18
LastEditTime: 2021-11-21 18:23:40
'''
from collections import deque

class A():
    def __init__(self , para):
        self.para = para
        self.buffer = deque(maxlen=2)
        print('this is init' , self.para)
    def test_fun(self):
        print('this is fun' , self.para)
        self.test_fun2()
    def test_fun2(self):
        print('this is fun2' , self.para)
    def loop(self , i):
        if i % 5 == 0 :
            self.buffer.append(i)
            print(self.buffer)
        if i % 5 == 0 :
            print('len:' , len(self.buffer))
            if len(self.buffer)>1:
                print('head' , self.buffer[len(self.buffer) - 1])
                if 10 in self.buffer and 15 in self.buffer:
                    print('find 10 and 15')
            # self.para -= 10
            # print('para changed ' , self.para)


