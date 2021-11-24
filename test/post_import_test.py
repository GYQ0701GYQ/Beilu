'''
Description: 
FilePath: /beilu/test/post_import_test.py
Autor: Rainche
Date: 2021-11-16 22:17:54
LastEditTime: 2021-11-21 17:14:20
'''
from post_test import A

def main(obj):
    obj.test_fun()
    print('hi rainche')
    # for i in range(16) :
    #     print('still in loop,i=' , i)
    #     obj.loop(i)

if __name__ == '__main__':
    obj = A(36)
    main(obj)
# lista = ['warning' , 'safe'] 
# taga = 'warning'
# tagb = 'safe'
# lista.remove(taga)
# print(lista[0])