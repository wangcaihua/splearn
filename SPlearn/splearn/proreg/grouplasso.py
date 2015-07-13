# Author: Caihua Wang <490419716@qq.com>

from numpy.linalg import norm
from numpy import zeros

def grouplasso(v, lam): 
	normv = norm(v)
	if normv <= lam:
		return zeros(len(v))
	else:
		return (1.0-lam/normv)*v 
