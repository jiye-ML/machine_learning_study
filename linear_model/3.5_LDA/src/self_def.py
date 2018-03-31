# -*- coding: utf-8 -*
import numpy as np


# 获得一个点到在一条线的投影
def GetProjectivePoint_2D(point, line):
    '''
    功能： 
    @param point:  [a,b]
    @param line: 直线 [k, t] which means y = k*x + t
    @return: 投影
    '''
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if k == 0:
        return [a, t]
    elif k == np.inf:
        return [0, b]
    # x 在直线上的投影
    x = (a+k*b-k*t) / (k*k+1)
    y = k*x + t
    return [x, y]

         
    
    
    