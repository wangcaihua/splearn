# Author: Caihua Wang <490419716@qq.com>

from numpy.linalg import norm
from numpy.random import uniform
from numpy import dot, array, sign, diag 

__all__ = ['cwd_lasso', 'cwd_elasticnet']

def cwd_lasso(X, y, lam, theta=None, maxiter=500, eps=1e-6):
	'''
		coordinate-wise descent algorithms for lasso 

		Parameters
		----------

		X : 2d numpy array, the input data 
		y : 1d numpy array, the input label 
		lam : float, a positive real, the regularization
		theta :  1d numpy array, the inital parameters 
		maxinter : int, the max iter times 
		eps : float, the error tolerance 

		Example
		----------
		>> from splearn import cwd_lasso
		>> from numpy.random import uniform, randn
		>> from numpy import dot 
		>> 
		>> X = uniform(-10, 10, size=(200, 50))
		>> theta0 = uniform(-20, 20, size=50)
		>> y = dot(X, theta0) + randn(200)
		>> 
		>> theta = cwd_lasso(X, y, lam=2)
		>>
	'''
	n, m = X.shape 
	if theta == None:
		theta = uniform(-0.01, 0.01, size=m)
	M, b = dot(X.T, X), dot(X.T, y)
	dM = diag(M)

	k = 0
	while k < maxiter:
		theta_old = theta.copy()
		for i in xrange(m):
			theta[i] = 0.0
			z = b[i] - dot(M[i], theta)
			dz = (abs(z) - lam)/dM[i]
			if dz > 0:
				theta[i] = sign(z) * dz

		if norm(theta_old-theta) < eps:
			break
		k += 1

	return theta

def cwd_elasticnet(X, y, lam1, lam2, theta=None, maxiter=500, eps=1e-6):
	'''
		coordinate-wise descent algorithms for elastic net

		Parameters
		----------

		X : 2d numpy array, the input data 
		y : 1d numpy array, the input label 
		lam1, lam2 : float, positive real, the regularization
		theta :  1d numpy array, the inital parameters 
		maxinter : int, the max iter times 
		eps : float, the error tolerance 

		Example
		----------
		>> from splearn import cwd_elasticnet
		>> from numpy.random import uniform, randn
		>> from numpy import dot 
		>> 
		>> X = uniform(-10, 10, size=(200, 50))
		>> theta0 = uniform(-20, 20, size=50)
		>> y = dot(X, theta0) + randn(200)
		>> 
		>> theta = cwd_elasticnet(X, y, lam1=0.8, lam2=1.0)
		>>
	'''
	n, m = X.shape 
	if theta == None:
		theta = uniform(-0.01, 0.01, size=m)
	M, b = dot(X.T, X), dot(X.T, y)
	dM = diag(M) + 2.0*lam2

	k = 0
	while k < maxiter:
		theta_old = theta.copy()
		for i in xrange(m):
			theta[i] = 0.0
			z = b[i] - dot(M[i], theta)
			dz = (abs(z) - lam1)/dM[i]
			if dz > 0:
				theta[i] = sign(z) * dz

		if norm(theta_old-theta) < eps:
			break
		k += 1

	return theta