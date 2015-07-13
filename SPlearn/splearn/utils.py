# Author: Caihua Wang <490419716@qq.com>

from cPickle import load
from numpy.linalg import norm 

def loadata(fname):	
	fid = open(fname, 'rb')
	diabetes = load(fid)
	fid.close()
	X = diabetes['data']
	y = diabetes['target']
	return X, y

def accmse(y, py, calmod='acc'):
	if calmod=='acc':
		cc = 0.0 
		for i, j in zip(y, py):
			if i == j:
				cc += 1.0 
		res = cc/len(y)
	elif calmod=='mse':
		res = 1.0 - (norm(y-py)/norm(y))**2
	else:
		print 'ERROR!'
		res = None
	return res 

def plotpath(coefs, pathtype):
	import matplotlib.pyplot as plt
	xx = abs(coefs).sum(axis=1)
	xx /= xx[-1]

	plt.plot(xx, coefs, '--s')
	ymin, ymax = plt.ylim()
	plt.vlines(xx, ymin, ymax, linestyle='dashed')
	plt.xlabel('|coef| / max|coef|')
	plt.ylabel('Coefficients')
	plt.title(pathtype +' Path')
	plt.axis('tight')
	plt.show()

