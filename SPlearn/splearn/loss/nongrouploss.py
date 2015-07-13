"""
Non-group loss functions.
"""

# Author: Caihua Wang <490419716@qq.com>

from numpy import exp, log, dot, array 

__all__ = ['squarederror', 'huberloss', 'modifiedhuberloss', 'logit', 'squaredhinge', 'huberizedhinge']


def squarederror(X, y, weight=None, fitintercept=True, alpha=None):
	n, p = X.shape 
	if weight!=None:
		assert len(weight) == n 
	intercept = y.mean() if fitintercept==True else None 

	def objfunc(theta):
		if fitintercept==True:
			tmp = abs(y - dot(X, theta) - intercept)**2
		else:
			tmp = abs(y - dot(X, theta))**2

		if weight != None:
			tmp *= weight
		return 0.5 * tmp.sum()

	def dfunc(theta):
		if fitintercept==True:
			tmp = dot(X, theta) + intercept - y
		else:
			tmp = dot(X, theta) - y

		if weight != None:
			tmp *= weight
		return dot(X.T, tmp) 

	return objfunc, dfunc, p, intercept

def huberloss(X, y, delta, weight=None, fitintercept=True, alpha=None):
	n, p = X.shape 
	if weight!=None:
		assert len(weight) == n 
	intercept = y.mean() if fitintercept==True else None 

	def objfunc(theta):
		if fitintercept==True:
			tmp = abs(dot(X, theta) + intercept - y)
		else:
			tmp = abs(dot(X, theta) - y)

		mask1 = tmp <= delta
		mask2 = tmp > delta
		tmp[mask1] = 0.5*tmp[mask1]**2
		tmp[mask2] = delta*(tmp[mask2]-0.5*delta)
		if weight != None:
			tmp *= weight
		return tmp.sum()

	def dfunc(theta):
		if fitintercept==True:
			tmp = dot(X, theta) + intercept - y
		else:
			tmp = dot(X, theta) - y

		tmp[tmp < -delta] = -delta
		tmp[tmp > delta] = delta
		if weight != None:
			tmp *= weight
		return dot(X.T, tmp)

	return objfunc, dfunc, p, intercept

def modifiedhuberloss(X, y, weight=None, fitintercept=True, alpha=0.05):
	n, p = X.shape 
	if weight!=None:
		assert len(weight) == n 
	if fitintercept==True:
		intercept = log(1.0*sum(y >= 0)/(sum(y < 0)))
		while True:
			tmp = y * intercept
			mask1 = array([-1 < val <=1 for val in tmp])
			mask2 = tmp <= -1
			mask3 = tmp <= 1
			tmp[mask1] = 2*(tmp[mask1]-1)*y[mask1]
			tmp[mask2] = -4*y[mask2]
			if weight != None:
				tmp[mask3] *= weight[mask3]
			alpha *= 0.9
			dfint = tmp[mask3].sum()
			if alpha * abs(dfint) < 1e-8:
				break
			else:
				intercept -= alpha * dfint
	else:
		intercept = None 

	def objfunc(theta):
		if fitintercept==True:
			tmp = y * (dot(X, theta)+intercept)
		else:
			tmp = y * dot(X, theta)

		mask1 = array([-1 < val <=1 for val in tmp])
		mask2 = tmp <= -1
		mask3 = tmp <= 1
		tmp[mask1] = (tmp[mask1]-1)**2
		tmp[mask2] = -4*tmp[mask2]
		if weight != None:
			tmp[mask3] *= weight[mask3]
		return tmp[mask3].sum()

	def dfunc(theta):
		if fitintercept==True:
			tmp = y * (dot(X, theta)+intercept)
		else:
			tmp = y * dot(X, theta)

		mask1 = array([-1 < val <=1 for val in tmp])
		mask2 = tmp <= -1
		mask3 = tmp <= 1
		tmp[mask1] = 2*(tmp[mask1]-1)*y[mask1]
		tmp[mask2] = -4*y[mask2]
		if weight != None:
			tmp[mask3] *= weight[mask3]
		return dot(X[mask3].T, tmp[mask3]) 

	return objfunc, dfunc, p, intercept

def logit(X, y, weight=None, fitintercept=True, alpha=None):
	n, p = X.shape 
	if weight!=None:
		assert len(weight) == n 
	if fitintercept==True:
		intercept = log(1.0*sum(y >= 0)/(sum(y < 0)))
	else:
		intercept = None 

	def objfunc(theta):
		if fitintercept==True:
			tmp = y * (dot(X, theta)+intercept).clip(-709, 709)
		else:
			tmp = y * dot(X, theta).clip(-709, 709)

		tmp = log(1.0+exp(-tmp))
		if weight != None:
			tmp *= weight
		return tmp.sum() 

	def dfunc(theta):
		if fitintercept==True:
			tmp = exp(-y*(dot(X, theta)+intercept).clip(-709, 709))
		else:
			tmp = exp(-y*dot(X, theta).clip(-709, 709))
		tmp = (-y*tmp)/(1.0+tmp)
		if weight != None:
			tmp *= weight
		return dot(X.T, tmp)

	return objfunc, dfunc, p, intercept

def squaredhinge(X, y, weight=None, fitintercept=True, alpha=0.01):
	n, p = X.shape 
	if weight!=None:
		assert len(weight) == n 
	if fitintercept==True:
		intercept = log(1.0*sum(y >= 0)/(sum(y < 0)))
		while True:
			tmp = 1 - y*intercept
			mask = tmp > 0
			tmp[mask] *= -y[mask]
			if weight != None:
				tmp[mask] *= weight[mask]
			alpha *= 0.9
			dfint = tmp[mask].sum()
			if alpha * abs(dfint) < 1e-8:
				break
			else:
				intercept -= alpha * dfint
	else:
		intercept = None 

	def objfunc(theta):
		if fitintercept==True:
			tmp = 1 - y * (dot(X, theta)+intercept)
		else:
			tmp = 1 - y * dot(X, theta)

		mask = tmp > 0
		tmp[mask] = tmp[mask]**2
		if weight != None:
			tmp[mask] *= weight[mask]
		return 0.5*tmp[mask].sum()

	def dfunc(theta):
		if fitintercept==True:
			tmp = y*(dot(X, theta)+intercept)
		else:
			tmp = y*dot(X, theta)

		mask = tmp <= 1
		tmp[mask] = y[mask]*(tmp[mask]-1)
		if weight != None:
			tmp[mask] *= weight[mask]
		return dot(X[mask].T, tmp[mask])

	return objfunc, dfunc, p, intercept

def huberizedhinge(X, y, delta, weight=None, fitintercept=True, alpha=1.0):
	n, p = X.shape 
	if weight!=None:
		assert len(weight) == n 
	if fitintercept==True:
		intercept = log(1.0*sum(y >= 0)/(sum(y < 0)))
		while True:
			tmp = y * intercept
			mask1 = array([1-delta < val <=1 for val in tmp])
			mask2 = tmp <= 1-delta
			mask3 = tmp <= 1
			tmp[mask1] = y[mask1]*(tmp[mask1]-1)/delta
			tmp[mask2] = -y[mask2]
			if weight != None:
				tmp[mask3] *= weight[mask3]
			alpha *= 0.9 
			dfint = tmp[mask3].sum()
			if alpha * abs(dfint) < 1e-8:
				break
			else:
				intercept -= alpha * dfint
		else:
			intercept = None 

	def objfunc(theta):
		if fitintercept==True:
			tmp = y * (dot(X, theta)+intercept)
		else:
			tmp = y * dot(X, theta)

		mask1 = array([1-delta < val <=1 for val in tmp])
		mask2 = tmp <= 1-delta
		mask3 = tmp <= 1
		tmp[mask1] = 0.5*(1-tmp[mask1])**2 /delta
		tmp[mask2] = 1 - delta/2 - tmp[mask2]
		if weight != None:
			tmp[mask3] *= weight[mask3]
		return tmp[mask3].sum()

	def dfunc(theta):
		if fitintercept==True:
			tmp = y * (dot(X, theta)+intercept)
		else:
			tmp = y * dot(X, theta)

		mask1 = array(map(lambda val: 1-delta < val <=1, tmp))
		mask2 = tmp <= 1-delta
		mask3 = tmp <=1
		tmp[mask1] = y[mask1]*(tmp[mask1]-1)/delta
		tmp[mask2] = -y[mask2]
		if weight != None:
			tmp[mask3] *= weight[mask3]
		return dot(X[mask3].T, tmp[mask3])

	return objfunc, dfunc, p, intercept

