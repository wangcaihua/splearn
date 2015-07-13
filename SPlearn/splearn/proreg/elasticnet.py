# Author: Caihua Wang <490419716@qq.com>

from numpy import sign 

def elasticnet(v, lam2, lam1): 
	res = abs(v) - lam1
	res[res<0] = 0
	return sign(v)*res/(1.0 + 2*lam2)