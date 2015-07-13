"""
Sparse sigular value decomposition
Sparse coding
"""

# Author: Caihua Wang <490419716@qq.com>

from numpy import dot, array
from numpy.linalg import norm
from ..proreg import (lasso,
					  elasticnet,
					  fusedlasso,
					  sparseness)

__all__ = ['ssvd', 'spcoding']

spprofunc = {'lasso':lasso,
			 'elasticnet':elasticnet, 
			 'fusedlasso':fusedlasso, 
			 'sparseness':sparseness,	
			 }


def _rank1(X, profunc1, args1, profunc2, args2, niter=500, eps=1e-6):
	u = X.sum(axis=1); u /= norm(u)
	v = X.sum(axis=0); v /= norm(v)

	rho_old = dot(dot(u,X),v)
	for i in xrange(niter):
		alpha = dot(X,v)
		if profunc1 == None:
			u = alpha
		else:
			u = profunc1(alpha, **args1)
		u /= norm(u)

		beta = dot(X.T,u)
		if profunc2 == None:
			v = beta
		else:
			v = profunc2(beta, **args2)
		rho = norm(v)
		v /= rho

		if abs(rho-rho_old) <= eps:
			break
		else:
			rho_old = rho

	return u, rho, v 

def ssvd(X, k, pfname1, args1, pfname2, args2):
	'''
		sparse sigular value decomposition

		Parameters
		----------

		X : 2d numpy array, the input matrix
		k : int, the number of svd laters
		pfname*, args*:
			pfname* : string, specify the regularization item, as follows:
				'lasso', 'elasticnet', 'fusedlasso', 'sparseness'
			args* : dictionary, specify the regularization supper parameters,
				'lasso': args = {'lam': val}, val > 0
				'sparseness' : args = {'lam': val}, val in (0, 1)
				'elasticnet', 'fusedlasso': args = {'lam1': val1, 'lam2':val2}
			* : 1, 2.
				1 --> U's row
				2 --> V's row

		Example
		----------

		>> from splearn import ssvd
		>> from numpy.random import uniform 
		>>
		>> X = uniform(-10, 10, size=(50, 80))
		>> args1 = {'lam':0.5}
		>> args2 = {'lam':0.7}
		>> U, D, V = ssvd(X, 10, pfname1='sparseness', args1=args1,
								 pfname2='sparseness', args2=args2)
		>> 

	'''
	assert pfname1 in spprofunc
	assert pfname2 in spprofunc
	U, D, V = [], [], []
	Xt = X.copy()
	if pfname1 == None:
		profunc1 = None
	else:
		profunc1 = spprofunc[pfname1]

	if pfname2 == None:
		profunc2 = None
	else:
		profunc2 = spprofunc[pfname2]

	for i in xrange(k):
		u, d, v = _rank1(Xt, profunc1, args1, profunc2, args2)
		Xt -= d*dot(u.reshape(-1,1), v.reshape(1,-1))
		U.append(u); D.append(d); V.append(v)

	return array(U), array(D), array(V)

def spcoding(X, k, pfname1, args1, pfname2, args2):
	'''
		sparse coding using sparse sigular value decomposition

		Parameters
		----------

		X : 2d numpy array, the input matrix
		k : int, the number of svd laters
		pfname*, args*:
			pfname* : string, specify the regularization item, as follows:
				'lasso', 'elasticnet', 'fusedlasso', 'sparseness'
			args* : dictionary, specify the regularization supper parameters,
				'lasso': args = {'lam': val}, val > 0
				'sparseness' : args = {'lam': val}, val in (0, 1)
				'elasticnet', 'fusedlasso': args = {'lam1': val1, 'lam2':val2}
			* : 1, 2.
				1 --> U's row, Note: U's row are unit norm 
				2 --> V's row

		Example
		----------

		>> from splearn import ssvd
		>> from numpy.random import uniform 
		>>
		>> X = uniform(-10, 10, size=(50, 80))
		>> args1 = {'lam':0.5}
		>> args2 = {'lam':0.7}
		>> U, V = ssvd(X, 10, pfname1='sparseness', args1=args1,
							  pfname2='sparseness', args2=args2)
		>> 

	'''
	assert pfname1 in spprofunc
	assert pfname2 in spprofunc
	U, D, V = ssvd(X=X, k=k, pfname1=pfname1, args1=args1, 
									 pfname2=pfname2, args2=args2)
	Unrm = array([norm(U[i]) for i in xrange(k)])
	U /= Unrm.reshape(-1,1)
	D *= Unrm
	V *= D.reshape(-1,1)
	base, code = U.T, V
	return base, code

