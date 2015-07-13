"""
Sparse principal component analysis
"""

# Author: Caihua Wang <490419716@qq.com>

from numpy import dot, zeros
from numpy.linalg import norm
from ..proreg import (lasso,
					  elasticnet,
					  fusedlasso,
					  sparseness)

__all__ = ['spca']


spprofunc = {'lasso':lasso,
			 'elasticnet':elasticnet, 
			 'fusedlasso':fusedlasso, 
			 'sparseness':sparseness,	
			 }

def _spc1(Xu, Xv, profunc, args, niter=500, err=1e-6):
	n, m = Xv.shape 
	u = Xv.sum(axis=1); u /= norm(u)
	v = Xv.sum(axis=0); v /= norm(v)
	ro_old = dot(dot(u,Xv), v)

	for i in range(niter):
		u = dot(Xu, v); u /= norm(u)

		v = profunc(dot(Xv.T, u), **args)
		ro = norm(v);  

		if abs(ro - ro_old) <= err:
			break 
		else:
			ro_old = ro 

	return u, v

def spca(X, k, pfname, args):
	'''
		sparse principal component analysis

		Parameters
		----------

		X : 2d numpy array, the input matrix
		k : int, the number of svd laters
		pfname, args:
			pfname : string, specify the regularization item, as follows:
				'lasso', 'elasticnet', 'fusedlasso', 'sparseness'
			args : dictionary, specify the regularization supper parameters,
				'lasso': args = {'lam': val}, val > 0
				'sparseness' : args = {'lam': val}, val in (0, 1)
				'elasticnet', 'fusedlasso': args = {'lam1': val1, 'lam2':val2}

		Example
		----------

		>> from splearn import spca
		>> from numpy.random import uniform 
		>>
		>> X = uniform(-10, 10, size=(50, 80))
		>> args1 = {'lam':0.5}
		>> args2 = {'lam':0.7}
		>> U, V = spca(X, 10, pfname1='sparseness', args1=args1,
							  pfname2='sparseness', args2=args2)
		>> 

	'''
	assert pfname in spprofunc
	n, m = X.shape 
	U, V = zeros((k,n)), zeros((k,m))
	profunc = spprofunc[pfname]

	for i in range(k):
		Xu = X - dot(dot(U.T, U), X)
		u, v = _spc1(Xu, X, profunc, args)
		U[i], V[i] = u, v
	return U, V 
