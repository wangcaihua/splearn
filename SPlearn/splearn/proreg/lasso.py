# Author: Caihua Wang <490419716@qq.com>

from numpy import sign 

def lasso(v, lam): 
	res = abs(v) - lam
	res[res<0] = 0
	return sign(v) * res