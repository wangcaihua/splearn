"""
Group sparse linear model.
"""

# Author: Caihua Wang <490419716@qq.com>


from numpy import dot, exp, array 

from ..loss import (SQUAREDERROR, 
					HUBERLOSS, 
					MODIFIEDHUBERLOSS, 
					LOGIT, 
					SQUAREDHINGE, 
					HUBERIZEDHINGE)

from ..proreg import (grouplasso, 
					  sparsegrouplasso)

from ..solver import gmd

__all__ = ['GSLM']

GroupedLoss = { 'squarederror':SQUAREDERROR,  
				'huberloss':HUBERLOSS, 
				'logit':LOGIT, 
				'modifiedhuberloss':MODIFIEDHUBERLOSS, 
				'huberizedhinge':HUBERIZEDHINGE,
				'squaredhinge':SQUAREDHINGE}

GroupedRegu = { 'grouplasso':grouplasso, 
				'sparsegrouplasso':sparsegrouplasso}

def _profunc(y, dfy, L, regu, args):
	u = y - dfy/L 
	targs = {}
	if regu == 'grouplasso':
		targs['lam'] = (args['groupweight'] * args['lam'])/L
	elif regu == 'sparsegrouplasso':
		targs['lam1'] = args['lam1']/L
		targs['lam2'] = (args['groupweight'] * args['lam2'])/L
	return GroupedRegu[regu](u, **targs)

# SNGLM: Non-Grouped Sparse Linear Model
class GSLM(object):
	"""
	GSLM: Grouped Sparse Linear Model 
		GSLM(loss, delta, regu, reargs, grouplist, groupweight, fitintercept=True, alpha=0.05)

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
		'grouplasso', 'sparsegrouplasso'

	reargs : dictionary
		specify supper parameters for different regularizations, 
		'grouplasso':
			reargs = {'lam': val}
		'sparsegrouplasso':
			reargs = {'lam1': val1, 'lam2': val2}

	grouplist : list of tuple (cannot be None)
		grouplist = [(0, 5), (5, 20), (20, 30), (30, 45)]
		start with 0, end with the number of predictors.
		the end point of last group must identity with the next group's start point.

	groupweight : list of float | None 
		if groupweight is not None, then the length of groupweight must equal to the length of grouplist.


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
	>> from splearn import GSLM
	>> from numpy.random import uniform, randn
	>> import numpy as np 
	>> 
	>> X = uniform(-10, 10, size=(500, 200))
	>> theta = uniform(-10, 10, size=200)
	>> y = np.dot(X, theta) + randn(200)
	>>
	>> reargs = {'lam': 4}
	>> grouplist = [(0, 20), (20, 50), (50, 80), (80, 120), (120, 155), (155, 187), (187, 200)]
	>> lm = GSLM(loss='squarederror', delta=None, regu='lasso', reargs=reargs,grouplist=grouplist, groupweight=None)
	>> lm.fit(X, y)
	>> py = lm.predict(X)
	>> coeff = lm.coeff_
	>> intercept = lm.intercept_
	>> 
	"""

	def __init__(self, loss, delta, regu, reargs, grouplist, groupweight, fitintercept=True, alpha=0.01):
		assert loss in GroupedLoss
		assert regu in GroupedRegu

		self.loss, self.regu = loss, regu
		self.grouplist, self.groupweight = grouplist, groupweight
		self.delta, self.args = delta, reargs
		self.fitintercept = fitintercept
		self.alpha = alpha 

	def fit(self, X, y, weight=None, theta=None):
		''' The training function.
				X : 2d numpy array, the input predictors
				y : 1d numpy array, the labels
				weight: the aeight of samples 
				theta: the inital parameters
		'''

		if self.loss in ['huberloss', 'huberizedhinge']:
			lossobj = GroupedLoss[self.loss](X=X, y=y, 
						delta=self.delta,
						grouplist=self.grouplist, 
						groupweight=self.groupweight, 
						sampleweight=weight,
						theta0=theta,
						fitintercept=self.fitintercept,
						alpha=self.alpha)
		else:
			lossobj = GroupedLoss[self.loss](X=X, y=y, 
						grouplist=self.grouplist, 
						groupweight=self.groupweight, 
						sampleweight=weight,
						theta0=theta,
						fitintercept=self.fitintercept,
						alpha=self.alpha)

		gmd(lossobj=lossobj, Profunc=_profunc, regu=self.regu, args=self.args)
		self.coeff_ = lossobj.theta0
		if self.fitintercept:
			self.intercept_ = lossobj.intercept	

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
				tmp = dot(X, self.coeff_) + self.intercept_
			else:
				tmp = dot(X, self.coeff_)
			pp = 1.0/(1.0+exp(-tmp))
			pn = 1.0 - pp 
			return array([pp, pn]).T
		else:
			print 'The loss function: squarederror, huberloss \
				   cannot calculate predict_proba!'


