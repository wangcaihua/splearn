"""
Non-negative matrix decomposition.
"""

# Author: Caihua Wang <490419716@qq.com>

from math import sqrt
from numpy.linalg import norm
from numpy import dot, log, array
from numpy.random import uniform


__all__ = ['snmf']


def _fistaH(X, B, H0, alpha, niter=20):
	t0 =1.0
	for i in xrange(niter):	
		Y = dot(B, H0)	
		H1 = H0*dot(B.T, X/Y)/(1+alpha)
		t1 = (1.0+sqrt(1+4*t0**2))/2.0
		H0 = H1 + (t0-1)/t1 * (H1-H0)
		t0 = t1
	return H1 

def _fistaB(X, B0, H, niter=20):
	t0=1.0	
	for i in xrange(niter):
		Y = dot(B0, H)
		B1 = B0*dot(X/Y, H.T)/H.sum(axis=1).reshape(1,-1)
		B1 /= B1.sum(axis=0).reshape(1,-1)
		t1 = (1.0+sqrt(1+4*t0**2))/2.0
		B0 = B1 + (t0-1)/t1 * (B1-B0)
		t0 = t1
	return B1 

def _rank1(X, eps=1e-6):
	n, m = X.shape
	u = uniform(-0.01, 0.01, n); u /= norm(u)
	v = uniform(-0.01, 0.01, m); v /= norm(v)
	obj = dot(dot(u, X), v)
	while True:
		v = dot(X.T, u); v /= norm(v)
		u = dot(X, v); ro = norm(u); u /= ro 
		if abs(ro - obj) < eps:
			break
		else: 
			obj = ro
	return u, ro, v

def _initBH(X, k):
	U, V = [], []
	Xt = X.copy()
	for i in xrange(k):
		u, d, v = _rank1(Xt)
		Xt -= d*dot(u.reshape(-1,1), v.reshape(1,-1))

		xp = u.copy(); xp[xp<0]=0; xpnrm = norm(xp)
		u[u>0]=0; xn = -u; xnnrm = norm(xn)
		yp = v.copy(); yp[yp<0]=0; ypnrm = norm(yp)
		v[v>0]=0; yn = -v; ynnrm = norm(yn)
		mp = xpnrm*ypnrm; mn = xnnrm*ynnrm
		if mp > mn:
			u = sqrt(d*mp)*xp/xpnrm
			v = sqrt(d*mp)*yp/ypnrm
		else:
			u = sqrt(d*mn)*xn/xnnrm
			v = sqrt(d*mn)*yn/ynnrm

		usum = u.sum(); u /= usum; v *= usum
		U.append(u); V.append(v)
	return array(U).T, array(V) 

def snmf(X, k, alpha, nsubiter=30, eps=1e-4):
	'''
	sparse non-nrgative matrix factorization (snmf)
	Parameters
	----------

	X : 2d numpy array, the input matrix
	k : int, the component you want to get 
	alpha : float, the regularization factor, must be positive

	Example
	----------

	>> from splearn import snmf
	>> from numpy.random import uniform 
	>> import numpy as np 
	>> X = uniform(0, 100, size=(125, 50))
	>> B, H = snmf(X, k=10, alpha=0.2)
	>> print X - np.dot(B, H)
	>> 
	'''
	
	B, H = _initBH(X, k)
	Y = dot(B, H)
	obj_old = (X*log(X/Y)-X+Y).mean() + alpha*H.mean()

	i = 0
	while True:
		H = _fistaH(X=X, B=B, H0=H, alpha=alpha, niter=nsubiter)
		B = _fistaB(X=X, B0=B, H=H, niter=nsubiter)

		if i % 3 == 0:
			Y = dot(B, H)
			obj = (X*log(X/Y)-X+Y).mean() + alpha*H.mean()
			if abs(obj - obj_old) < eps:
				break
			else:
				obj_old = obj 
		i += 1
	return B, H 

