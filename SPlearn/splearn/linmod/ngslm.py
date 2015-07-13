"""
Non-group sparse linear model.
"""

# Author: Caihua Wang <490419716@qq.com>

from numpy import dot, exp, array 

from ..loss import (squarederror, 
					huberloss, 
					logit, 
					modifiedhuberloss, 
					huberizedhinge,
					squaredhinge)

from ..proreg import (lasso, 
					  elasticnet, 
					  fusedlasso)

from ..solver import fista 

__all__ = ['NGSLM']

NonGroupedLoss = {'squarederror':squarederror, 
				  'huberloss':huberloss, 
				  'logit':logit, 
				  'modifiedhuberloss':modifiedhuberloss, 
				  'huberizedhinge':huberizedhinge, 
				  'squaredhinge':squaredhinge}

NonGroupedRegu = {'lasso':lasso, 
				  'elasticnet':elasticnet, 
				  'fusedlasso':fusedlasso}

def _profunc(y, dfy, L, regu, args):
	u = y - dfy/L 
	targs = {}
	if 'lam' in args: targs['lam'] = args['lam']/L
	if 'lam1' in args: targs['lam1'] = args['lam1']/L
	if 'lam2' in args: targs['lam2'] = args['lam2']/L

	return NonGroupedRegu[regu](u, **targs)

# SNGLM: Non-Grouped Sparse Linear Model
class NGSLM(object):
	"""NGSLM: Non-Grouped Sparse Linear Model 
		NGSLM(loss, delta, regu, reargs, fitintercept=True, alpha=0.05)

	Parameters
	----------
	loss : string
		specify loss function by a string. loss must be one of the following:
		Regression:
			'squarederror'
			'huberloss'
		Classification:
			'logit'
			'modifiedhuberloss'
			'huberizedhinge'
			'squaredhinge'

	delta : float | None
		the parameter in loss function, such as 'huberloss' and 'huberizedhinge'.
		'huberloss' : delta in [0, inf], usually not very big, say smaller than 10.
		'huberizedhinge' : delta in (0, 1), usually close to 1, say 0.8.
		For other loss functions, delta is not used, so pls set 'delta = None'.

	regu : string
		specify regularization by a string. regu must be one of the following:
		'lasso', 'elasticnet', 'fusedlasso'

	reargs : dictionary
		specify supper parameters for different regularizations, 
		'lasso':
			reargs = {'lam': val}
		'elasticnet', 'fusedlasso':
			reargs = {'lam1': val1, 'lam2': val2}

	fitintercept : bool
		whether fit intercept.

	alpha : float
		if fit intercept, the parameter alpha is the learning rate for fitting intercept, so alpha should not too large. usually alpha in [0.01, 1.0].

	Attributes
	----------
	`coeff_` : the model parameter, a 1d numpy array.

	`intercept_` : intercept, a float or None.

	Example
	----------
	>> from splearn import NGSLM
	>> from numpy.random import uniform, randn
	>> import numpy as np 
	>> 
	>> X = uniform(-10, 10, size=(500, 200))
	>> theta = uniform(-10, 10, size=200)
	>> y = np.dot(X, theta) + randn(200)
	>>
	>> reargs = {'lam': 100}
	>> lm = NGSLM(loss='squarederror', delta=None, regu='lasso', reargs=reargs)
	>> lm.fit(X, y)
	>> py = lm.predict(X)
	>> coeff = lm.coeff_
	>> intercept = lm.intercept_
	>> 
	"""

	def __init__(self, loss='squarederror', delta=None, regu='lasso', reargs=None, fitintercept=True, alpha=0.05):
		assert loss in NonGroupedLoss
		assert regu in NonGroupedRegu

		self.loss, self.regu = loss, regu
		self.delta, self.args = delta, reargs
		self.fitintercept, self.alpha = fitintercept, alpha

	def fit(self, X, y, weight=None, theta=None):
		''' The training function.
				X : 2d numpy array, the input predictors
				y : 1d numpy array, the labels
				weight: the aeight of samples 
				theta: the inital parameters
		'''

		if self.loss in ['huberloss', 'huberizedhinge']:
			objfunc, dfunc, p, intercept = NonGroupedLoss[self.loss](
								X=X, y=y, 
								delta=self.delta, weight=weight,
								fitintercept=self.fitintercept,
								alpha=self.alpha)
		else:
			objfunc, dfunc, p, intercept = NonGroupedLoss[self.loss](
									X=X, y=y,
									weight=weight,
									fitintercept=self.fitintercept,
									alpha=self.alpha)

		self.coeff_ = fista(objfunc=objfunc, dfunc=dfunc, p=p, 
							Profunc=_profunc, regu=self.regu, args=self.args, 
							theta=theta, eta=2, L0=1.0, maxiter=500, eps=1e-6)
		self.intercept_ = intercept

	def predict(self, X):
		'''
			X : 2d numpy array, the input predictors
		'''

		if self.fitintercept:
			res = dot(X, self.coeff_) + self.intercept_
		else:
			res = dot(X, self.coeff_)

		if self.loss in ['squarederror', 'huberloss']:
			return res
		else:
			res[res>=0] = 1.0
			res[res< 0] = -1.0
			return res 

	def predict_proba(self, X):
		'''
			X : 2d numpy array, the input predictors
		'''
		
		if self.loss not in ['squarederror', 'huberloss']:
			if self.fitintercept:
				tmp = dot(X, self.coeff_) + self.intercept
			else:
				tmp = dot(X, self.coeff_)

			pp = 1.0/(1.0+exp(-tmp))
			pn = 1.0 - pp 
			return array([pp, pn]).T
		else:
			print 'The loss function: squarederror, huberloss \
				   cannot calculate predict_proba!'


