# Author: Caihua Wang <490419716@qq.com>

from numpy.linalg import norm
from numpy import zeros, sign

def sparsegrouplasso(v, lam2, lam1): 
	if max(abs(v)) <= lam1:
		return zeros(len(v))
	else:
		tmp = abs(v) - lam1
		tmp[tmp<0] = 0
		normv = norm(tmp)

		if normv <= lam2:
			return zeros(len(v))
		else:
			return sign(v)*(1-lam2/normv)*tmp
