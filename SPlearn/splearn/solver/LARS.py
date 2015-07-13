# Author: Caihua Wang <490419716@qq.com>

from math import sqrt
from numpy.linalg import inv
from numpy import sign, zeros, dot, inf, argmax, array

__all__ = ['lars_path', 'lars']

def _absmax(x):
	i = argmax(abs(x))
	val = abs(x[i])
	return i, val

def _min_pos1(C, c, A, a, inactive):
	res = inf 
	for i in inactive:
		r1 = (C + c[i])/(A + a[i])
		r2 = (C - c[i])/(A - a[i])
		if r1 <= 0 and r2 <= 0:
			continue
		elif r1 <= 0 and r2 > 0:
			r = r2
		elif r1 > 0 and r2 <= 0:
			r = r1
		else:
			r = min(r1, r2)

		if r < res:
			idx, res = i, r
	return idx, res

def _min_pos2(beta, s, w, active):
	idx, res = -1, inf
	d = - beta[active]/(s*w)
	td = d[d>0]
	if len(td) > 0:
		res = td.min()
		idx = active[d.tolist().index(res)]
	return idx, res

def lars_path(X, y, method='lar'):
	'''
		create path for lasso and lar algorithm

		Parameters
		----------
		X : 2d numpy array, the input data 
		y : 1d numpy array, the input label 
		method : 'lar'|'lasso'
			specify the path type, 'lar' or 'lasso'

		Example
		----------
		>> from splearn import lars_path, plotpath
		>> from numpy.random import uniform, randn
		>> from numpy import dot 
		>> 
		>> X = uniform(-10, 10, size=(200, 50))
		>> theta0 = uniform(-20, 20, size=50)
		>> y = dot(X, theta0) + randn(200)
		>> 
		>> path = lars_path(X, y, method='lasso')
		>> plotpath(path, pathtype='LASSO')
		>> 

	'''
	assert method in ('lar', 'lasso')
	n, m = X.shape 
	path = []
	Cov = dot(X.T, y)
	i, C = _absmax(Cov)
	active = [i]
	inactive = [j for j in range(m) if j != i]
	beta = zeros(m)
	path.append(beta.copy())
	
	while True:
		s = sign(Cov)[active]
		Xa = X[:,active] * s
		Ga_inv = inv(dot(Xa.T, Xa))
		A = 1.0/sqrt(Ga_inv.sum())
		w = A * Ga_inv.sum(axis=1)
		u = dot(Xa, w)
		a = dot(X.T, u)

		if inactive:
			idx1, r1 = _min_pos1(C, Cov, A, a, inactive)
			out = False
		else:
			r1 = C/A 
			out = True 

		drop, r = False, r1
		if method == 'lasso':
			idx2, r2 = _min_pos2(beta, s, w, active)
			if idx2 != -1 and r2 < r1:
				r = r2
				drop = True

		#updata beta
		beta[active] += r*s*w
		path.append(beta.copy())

		# update active, incative
		if drop:
			inactive.append(idx2)
			active.remove(idx2)
		elif inactive:
			active.append(idx1)
			inactive.remove(idx1)
		
		# update Cov, C
		Cov -= r*a
		C = max(abs(Cov)) if drop else C - r*A

		# loop control
		if out and not inactive: break

	return array(path)

def lars(X, y, lam=None, method='lasso'):
	'''
		LARS for lasso and lar

		Parameters
		----------
		X : 2d numpy array, the input data 
		y : 1d numpy array, the input label 
		lam : float | None, the regularization constant
		method : 'lar'|'lasso'
			specify the path type, 'lar' or 'lasso'

		Example
		----------
		>> from splearn import lars
		>> from numpy.random import uniform, randn
		>> from numpy import dot 
		>> 
		>> X = uniform(-10, 10, size=(200, 50))
		>> theta0 = uniform(-20, 20, size=50)
		>> y = dot(X, theta0) + randn(200)
		>> 
		>> beta = lars_path(X, y, lam=10, method='lasso')
		>> 

	'''
	assert method in ('lar', 'lasso')
	n, m = X.shape 
	path = []
	Cov = dot(X.T, y)
	i, C = _absmax(Cov)
	active = [i]
	inactive = [j for j in range(m) if j != i]
	beta = zeros(m)
	if lam != None:
		assert 0 <= lam < C
	
	while True:
		s = sign(Cov)[active]
		Xa = X[:,active] * s
		Ga_inv = inv(dot(Xa.T, Xa))
		A = 1.0/sqrt(Ga_inv.sum())
		w = A * Ga_inv.sum(axis=1)
		u = dot(Xa, w)
		a = dot(X.T, u)

		if inactive:
			idx1, r1 = _min_pos1(C, Cov, A, a, inactive)
			out = False
		else:
			r1 = C/A 
			out = True 

		drop, r = False, r1
		if method == 'lasso':
			idx2, r2 = _min_pos2(beta, s, w, active)
			if idx2 != -1 and r2 < r1:
				r = r2
				drop = True

		if lam != None and C - r*A < lam < C:
			r = (C - lam)/A
			beta[active] += r*s*w
			return beta

		#updata beta
		beta[active] += r*s*w

		# update active, incative
		if drop:
			inactive.append(idx2)
			active.remove(idx2)
		elif inactive:
			active.append(idx1)
			inactive.remove(idx1)
		
		# update Cov, C
		Cov -= r*a
		C = max(abs(Cov)) if drop else C - r*A

		# loop control
		if out and not inactive: break

	return beta
