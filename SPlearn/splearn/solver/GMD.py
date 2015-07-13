# Author: Caihua Wang <490419716@qq.com>

from .FISTA import fista 
from numpy.linalg import norm


__all__ = ['gmd']


def gmd(lossobj, Profunc, regu, args, eps=1e-6):
	lam = args['lam'] if 'lam' in args else args['lam2']

	# find the lam0
	last, lams = False, []
	for i in xrange(len(lossobj)):
		objfunc, dfunc, p, w = lossobj.createfuncs(i)
		theta = lossobj.getparm(i)
		lams.append(norm(dfunc(theta))/w)
	lam0 = max(lams); idx0 = lams.index(lam0); lams[idx0] = 0
	lam1 = max(lams); idx1 = lams.index(lam1); lams[idx1] = 0
	lam2 = max(lams); idx2 = lams.index(lam2); lams[idx2] = 0
	lams[idx0], lams[idx1], lams[idx2] = lam0, lam1, lam2
	
	if lam > lam0:
		print 'lam or lam2: must smaller than %f !' % lam0
	assert lam < lam0

	culam = 0.7*lam1 + 0.3*lam2
	if culam <= lam:
		culam = lam
		last = True

	# create set
	S, Sc = [], []
	for i, val in enumerate(lams):
		if val >= 2*culam-lam0:
			S.append(i)
		else:
			Sc.append(i)

	if len(Sc) == 0:
		last = True 
		culam = lam 
	else:
		if 'lam' in args:
			args['lam'] = culam
		else:
			args['lam2'] = culam

	while True:		
		# optmimize in set S
		loss_old = lossobj.calculateloss(args)
		while True:
			for i in S:
				objfunc, dfunc, p, w = lossobj.createfuncs(i)
				theta0 = lossobj.getparm(i)
				args['groupweight'] = w
				theta = fista(objfunc=objfunc, dfunc=dfunc, p=p, Profunc=Profunc, regu=regu, args=args, theta=theta0)
				lossobj.update(theta, i)
			loss = lossobj.calculateloss(args)
			if abs(loss-loss_old) < eps:
				if  'groupweight' in args:
					del args['groupweight']
				break
			else:
				loss_old = loss 

		# check in set Sc
		flag, lamsidx, lamsval = False, [], []
		for i in Sc:
			objfunc, dfunc, p, w = lossobj.createfuncs(i)
			theta = lossobj.getparm(i)
			val = norm(dfunc(theta))/w
			if val > culam:
				Sc.remove(i)
				S.append(i)
				flag = True
			else:
				lamsidx.append(i)
				lamsval.append(val)

		# renew the current lam and the set S, Sc
		if flag:
			if len(Sc) == 0:
				last = True 
				culam = lam 
				if 'lam' in args:
					args['lam'] = lam
				else:
					args['lam2'] = lam
		else:
			if last:
				break
			else:
				lam0 = max(lamsval); idx0 = lamsval.index(lam0); lamsval[idx0] = 0
				lam1 = max(lamsval); idx1 = lamsval.index(lam1); lamsval[idx1] = 0
				lamsval[idx0], lamsval[idx1] = lam0, lam1
				culam = 0.7*lam0+0.3*lam1
				if culam <= lam:
					culam = lam 
					last = True 

				if 'lam' in args:
					args['lam'] = culam
				else:
					args['lam2'] = culam

				for i, val in zip(lamsidx, lamsval):
					if val > culam:
						Sc.remove(i)
						S.append(i)


'''
def gmd(lossobj, Profunc, regu, args, eps=1e-6):
	while True:
		loss_old = lossobj.calculateloss(args)
		for i in xrange(len(lossobj)):
			objfunc, dfunc, p, w = lossobj.createfuncs(i)
			theta0 = lossobj.getparm(i)
			args['groupweight'] = w
			theta = fista(objfunc=objfunc, dfunc=dfunc, p=p, Profunc=Profunc, regu=regu, args=args, theta=theta0)
			lossobj.update(theta, i)
		loss = lossobj.calculateloss(args)
		if abs(loss-loss_old) < eps:
			break
		else:
			loss_old = loss

	del args['groupweight']
'''