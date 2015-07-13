"""
Group loss functions.
"""

# Author: Caihua Wang <490419716@qq.com>

from math import sqrt
from numpy.linalg import norm 
from numpy.random import uniform
from numpy import exp, log, dot, array, zeros


__all__ = ['SQUAREDERROR', 'HUBERLOSS', 'MODIFIEDHUBERLOSS', 'LOGIT', 'SQUAREDHINGE', 'HUBERIZEDHINGE']

class LOSS(object):
	def __init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=True):
		self.n, self.p = X.shape 
		self.X, self.y = X, y 
		self.grouplist = grouplist
		self.sampleweight = sampleweight
		self.ngroup = len(grouplist)
		self.fitintercept = fitintercept

		# check
		if theta0 == None:
			if fitintercept:
				self.theta0 = zeros(self.p)
			else:
				self.theta0 = uniform(-0.01, 0.01, self.p)
		else:
			assert len(theta0) == self.p
			self.theta0 = theta0
		
		if groupweight == None:
			groupweight = []
			for srart, stop in grouplist:
				groupweight.append(sqrt(1.0*(stop-srart)))
			self.groupweight = array(groupweight)
		else:
			assert len(grouplist) == len(groupweight)
			self.groupweight = groupweight

		if sampleweight != None:
			assert self.n == len(sampleweight)

		for i, (start, stop) in enumerate(grouplist):
			if i == 0:
				assert start == 0
				laststop = stop
			elif i == len(grouplist)-1:
				assert start == laststop
				assert stop == self.p
			else:
				assert start == laststop
				laststop = stop

	def __len__(self):
		return 	self.ngroup

	def calculateloss(self, args):
		if 'lam' in args:
			lam = args['lam']
			res = 0
			for i in xrange(self.ngroup):
				theta = self.getparm(i)
				res += lam * self.groupweight[i] * norm(theta,2)
			return res 
		elif 'lam1' in args and 'lam2' in args:
			lam1 = args['lam1']
			lam2 = args['lam2']
			res = 0
			for i in xrange(self.ngroup):
				theta = self.getparm(i)
				res += lam2 * self.groupweight[i] * norm(theta,2)
			return res + lam1 * norm(self.theta0,1)
		else:
			print "Error args!"
			return 0

	def getparm(self, k):
		start, stop = self.grouplist[k]
		return self.theta0[start:stop]

	def getnumparm(self,k):
		start, stop = self.grouplist[k]
		return stop - start

	def update(self, theta, k):
		start, stop = self.grouplist[k]
		self.theta0[start:stop] = theta


class SQUAREDERROR(LOSS):
	def __init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=True, alpha=None):
		LOSS.__init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=fitintercept)
		if fitintercept:
			self.intercept = y.mean()
		else:
			self.intercept = None

	def calculateloss(self, args):
		reg = LOSS.calculateloss(self, args)
		if self.fitintercept:
			tmp = (self.y - dot(self.X, self.theta0) - self.intercept)**2
		else:
			tmp = (self.y - dot(self.X, self.theta0))**2
		if self.sampleweight != None:
			tmp *= self.sampleweight

		return 0.5*tmp.sum() + reg

	def createfuncs(self, k):
		start, stop = self.grouplist[k]
		p = stop - start
		w = self.groupweight[k]
		theta_t = self.theta0.copy()
		theta_t[start:stop] = 0.0
		Xt = self.X[:,start:stop]
		yt = self.y - dot(self.X, theta_t)
		if self.fitintercept:
			yt -= self.intercept

		def objfunc(theta):
			tmp = (yt - dot(Xt, theta))**2
			if self.sampleweight != None:
				tmp *= self.sampleweight

			return 0.5*tmp.sum()

		def dfunc(theta):
			tmp = dot(Xt, theta) - yt
			if self.sampleweight != None:
				tmp *= self.sampleweight

			return dot(Xt.T, tmp)

		del theta_t
		return objfunc, dfunc, p, w 


class HUBERLOSS(LOSS):
	def __init__(self, X, y, delta, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=True, alpha=None):
		LOSS.__init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=fitintercept)
		self.delta = delta
		if fitintercept:
			self.intercept = y.mean()
		else:
			self.intercept = None

	def calculateloss(self, args):
		reg = LOSS.calculateloss(self, args)
		if self.fitintercept:
			tmp = abs(self.y - dot(self.X, self.theta0) - self.intercept)
		else:
			tmp = abs(self.y - dot(self.X, self.theta0))
		mask1 = tmp <= self.delta
		mask2 = tmp > self.delta
		tmp[mask1] = 0.5*tmp[mask1]**2
		tmp[mask2] = self.delta*(tmp[mask2]-0.5*self.delta)
		if self.sampleweight != None:
			tmp *= self.sampleweight

		return tmp.sum() + reg 

	def createfuncs(self, k):
		start, stop = self.grouplist[k]
		p = stop - start
		w = self.groupweight[k]
		theta_t = self.theta0.copy()
		theta_t[start:stop] = 0.0
		Xt = self.X[:,start:stop]
		yt = self.y - dot(self.X, theta_t)
		if self.fitintercept:
			yt -= self.intercept

		def objfunc(theta):
			tmp = abs(yt - dot(Xt, theta))
			mask1 = tmp <= self.delta
			mask2 = tmp > self.delta

			tmp[mask1] = 0.5*tmp[mask1]**2
			tmp[mask2] = self.delta*(tmp[mask2]-0.5*self.delta)
			if self.sampleweight != None:
				tmp *= self.sampleweight

			return tmp.sum()

		def dfunc(theta):
			tmp = dot(Xt, theta) - yt
			tmp[tmp < -self.delta] = -self.delta
			tmp[tmp > self.delta] = self.delta
			if self.sampleweight != None:
				tmp *= self.sampleweight

			return dot(Xt.T, tmp)

		del theta_t
		return objfunc, dfunc, p, w 


class MODIFIEDHUBERLOSS(LOSS):
	def __init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=True, alpha=0.05):
		LOSS.__init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=fitintercept)
		if fitintercept:
			self.intercept = log(1.0*sum(y>=0)/(sum(y<0)))			
			while True:
				tmp = self.y * self.intercept
				mask1 = array([-1 < val <=1 for val in tmp])
				mask2 = tmp <= -1
				mask3 = tmp <= 1
				tmp[mask1] = 2*(tmp[mask1]-1)*self.y[mask1]
				tmp[mask2] = -4*self.y[mask2]
				if self.sampleweight != None:
					tmp[mask3] *= self.sampleweight[mask3]
				alpha *= 0.9
				dfint = tmp[mask3].sum()
				if alpha * abs(dfint) < 1e-8:
					break
				else:
					self.intercept -= alpha * dfint
		else:
			self.intercept = None 

	def calculateloss(self, args):
		reg = LOSS.calculateloss(self, args)
		if self.fitintercept:
			tmp = self.y * (dot(self.X, self.theta0) + self.intercept)
		else:
			tmp = self.y * dot(self.X, self.theta0)
		mask1 = array([-1 < val <=1 for val in tmp])
		mask2 = tmp <= -1
		mask3 = tmp <= 1
		tmp[mask1] = (tmp[mask1]-1)**2
		tmp[mask2] = -4*tmp[mask2]
		if self.sampleweight != None:
			tmp[mask3] *= self.sampleweight[mask3]

		return tmp[mask3].sum() + reg 

	def createfuncs(self, k):
		start, stop = self.grouplist[k]
		p = stop - start
		w = self.groupweight[k]
		theta_t = self.theta0.copy()
		theta_t[start:stop] = 0.0
		Xt = self.X[:,start:stop]
		fixed = dot(self.X, theta_t)
		if self.fitintercept:
			fixed += self.intercept

		def objfunc(theta):
			tmp = self.y * (fixed + dot(Xt,theta))
			mask1 = array([-1 < val <=1 for val in tmp])
			mask2 = tmp <= -1
			mask3 = tmp <= 1
			tmp[mask1] = (tmp[mask1]-1)**2
			tmp[mask2] = -4*tmp[mask2]
			if self.sampleweight != None:
				tmp[mask3] *= self.sampleweight[mask3]

			return tmp[mask3].sum()

		def dfunc(theta):
			tmp = self.y * (fixed + dot(Xt,theta))
			mask1 = array([-1 < val <=1 for val in tmp])
			mask2 = tmp <= -1
			mask3 = tmp <= 1
			tmp[mask1] = 2*(tmp[mask1]-1)*self.y[mask1]
			tmp[mask2] = -4*self.y[mask2]
			if self.sampleweight != None:
				tmp[mask3] *= self.sampleweight[mask3]

			return dot(Xt[mask3].T, tmp[mask3])

		del theta_t
		return objfunc, dfunc, p, w  


class LOGIT(LOSS):
	def __init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=True, alpha=None):
		LOSS.__init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=fitintercept)
		if fitintercept:
			self.intercept = log(1.0*sum(y>=0)/(sum(y<0)))
		else:
			self.intercept = None 

	def calculateloss(self, args):
		reg = LOSS.calculateloss(self, args)
		if self.fitintercept:
			tmp = self.y*(dot(self.X, self.theta0) + self.intercept).clip(-709, 709)
		else:
			tmp = self.y*dot(self.X, self.theta0).clip(-709, 709)
		tmp = log(1.0+exp(-tmp))
		if self.sampleweight != None:
			tmp *= self.sampleweight

		return tmp.sum() + reg 

	def createfuncs(self, k):
		start, stop = self.grouplist[k]
		p = stop - start
		w = self.groupweight[k]
		theta_t = self.theta0.copy()
		theta_t[start:stop] = 0.0
		Xt = self.X[:,start:stop]
		fixed = dot(self.X, theta_t)
		if self.fitintercept:
			fixed += self.intercept

		def objfunc(theta):
			tmp = self.y*(fixed+dot(Xt, theta)).clip(-709,709)
			tmp = log(1.0+exp(-tmp))
			if self.sampleweight != None:
				tmp *= self.sampleweight

			return tmp.sum()

		def dfunc(theta):
			tmp = exp(-self.y*(fixed+dot(Xt, theta)).clip(-709,709))
			tmp = -(self.y*tmp)/(1.0+tmp)
			if self.sampleweight != None:
				tmp *= self.sampleweight

			return dot(Xt.T, tmp) 

		del theta_t
		return objfunc, dfunc, p, w 


class SQUAREDHINGE(LOSS):
	def __init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=True, alpha=0.01):
		LOSS.__init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=fitintercept)
		if fitintercept:
			self.intercept = log(1.0*sum(y>=0)/(sum(y<0)))			
			while True:
				tmp = 1 - self.y*self.intercept
				mask = tmp > 0
				tmp[mask] *= -self.y[mask]
				if self.sampleweight != None:
					tmp[mask] *= self.sampleweight[mask]
				alpha *= 0.9
				dfint = tmp[mask].sum()
				if alpha * abs(dfint) < 1e-8:
					break
				else:
					self.intercept -= alpha * dfint
		else:
			self.intercept = None 

	def calculateloss(self, args):
		reg = LOSS.calculateloss(self, args)
		if self.fitintercept:
			tmp = 1 - self.y*(dot(self.X, self.theta0) + self.intercept)
		else:
			tmp = 1 - self.y*dot(self.X, self.theta0)
		mask = tmp > 0
		tmp[mask] = tmp[mask]**2
		if self.sampleweight != None:
			tmp[mask] *= self.sampleweight[mask]

		return 0.5*tmp[mask].sum() + reg 

	def createfuncs(self, k):
		start, stop = self.grouplist[k]
		p = stop - start
		w = self.groupweight[k]
		theta_t = self.theta0.copy()
		theta_t[start:stop] = 0.0
		Xt = self.X[:,start:stop]
		fixed = dot(self.X, theta_t)
		if self.fitintercept:
			fixed += self.intercept

		def objfunc(theta):
			tmp = 1 - self.y*(fixed+dot(Xt, theta))
			mask = tmp > 0
			tmp[mask] = tmp[mask]**2
			if self.sampleweight != None:
				tmp[mask] *= self.sampleweight[mask]

			return 0.5*tmp[mask].sum()

		def dfunc(theta):
			tmp = 1 - self.y*(fixed+dot(Xt, theta))
			mask = tmp > 0
			tmp[mask] *= -self.y[mask]
			if self.sampleweight != None:
				tmp[mask] *= self.sampleweight[mask]

			return dot(Xt[mask].T, tmp[mask])

		del theta_t
		return objfunc, dfunc, p, w  


class HUBERIZEDHINGE(LOSS):
	def __init__(self, X, y, delta, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=True, alpha=1.0):
		LOSS.__init__(self, X, y, grouplist, groupweight=None, sampleweight=None, theta0=None, fitintercept=fitintercept)
		self.delta = delta 
		if fitintercept:
			self.intercept = log(1.0*sum(y>=0)/(sum(y<0)))		
			while True:
				tmp = self.y * self.intercept
				mask1 = array([1-self.delta < val <=1 for val in tmp])
				mask2 = tmp <= 1-self.delta
				mask3 = tmp <= 1
				tmp[mask1] = self.y[mask1]*(tmp[mask1]-1)/self.delta
				tmp[mask2] = -self.y[mask2]
				if self.sampleweight != None:
					tmp[mask3] *= self.sampleweight[mask3]
				alpha *= 0.9 
				dfint = tmp[mask3].sum()
				if alpha * abs(dfint) < 1e-8:
					break
				else:
					self.intercept -= alpha * dfint
		else:
			self.intercept = True

	def calculateloss(self, args):
		reg = LOSS.calculateloss(self, args)
		if self.fitintercept:
			tmp = self.y * (dot(self.X, self.theta0) + self.intercept)
		else:
			tmp = self.y * dot(self.X, self.theta0)
		mask1 = array([1-self.delta < val <=1 for val in tmp])
		mask2 = tmp <= 1-self.delta
		mask3 = tmp <= 1
		tmp[mask1] = 0.5*(1-tmp[mask1])**2 /self.delta
		tmp[mask2] = 1 - self.delta/2 - tmp[mask2]
		if self.sampleweight != None:
			tmp[mask3] *= self.sampleweight[mask3]

		return tmp[mask3].sum() + reg 

	def createfuncs(self, k):
		start, stop = self.grouplist[k]
		p = stop - start
		w = self.groupweight[k]
		theta_t = self.theta0.copy()
		theta_t[start:stop] = 0.0
		Xt = self.X[:,start:stop]
		fixed = dot(self.X, theta_t)
		if self.fitintercept:
			fixed += self.intercept

		def objfunc(theta):
			tmp = self.y * (fixed+dot(Xt, theta))
			mask1 = array([1-self.delta < val <=1 for val in tmp])
			mask2 = tmp <= 1-self.delta
			mask3 = tmp <= 1
			tmp[mask1] = 0.5*(1-tmp[mask1])**2 /self.delta
			tmp[mask2] = 1 - self.delta/2 - tmp[mask2]
			if self.sampleweight != None:
				tmp[mask3] *= self.sampleweight[mask3]

			return tmp[mask3].sum()

		def dfunc(theta):
			tmp = self.y * (fixed+dot(Xt, theta))
			mask1 = array([1-self.delta < val <=1 for val in tmp])
			mask2 = tmp <= 1-self.delta
			mask3 = tmp <= 1
			tmp[mask1] = self.y[mask1]*(tmp[mask1]-1)/self.delta
			tmp[mask2] = -self.y[mask2]
			if self.sampleweight != None:
				tmp[mask3] *= self.sampleweight[mask3]

			return dot(Xt[mask3].T, tmp[mask3])

		del theta_t
		return objfunc, dfunc, p, w  

