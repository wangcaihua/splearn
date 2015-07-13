# Author: Caihua Wang <490419716@qq.com>

from numpy.linalg import norm 
from numpy.random import uniform 
from numpy import array, zeros, ones, sign, dot

__all__ = ['fusedlasso']

def Rose(u):
	a = 0.0
	for idx, val in enumerate(u):
		a += (idx+1)*val
	a /= - (len(u) + 1)

	res = []
	for idx, val in enumerate(u[::-1]):
		if idx == 0:
			res.append(val + a)
		else:
			res.append(res[-1]+val)

	res.reverse()
	for i in xrange(1,len(res)):
		res[i] += res[i-1]

	return array(res) 

def Pro(vect, alpha):
	vect[vect >  alpha] =  alpha
	vect[vect < -alpha] = -alpha
	return vect 

def RRTx(z):
	res = []
	for i in xrange(len(z)):
		if i == 0:
			res.append(2*z[i]-z[i+1])
		elif i == len(z) -1:
			res.append(2*z[i]-z[i-1])
		else:
			res.append(2*z[i]-z[i+1]-z[i-1])
	return array(res)

def w(z, v, grad, lam2):
	x = zeros(len(v))
	Gset = []
	idx_old = 0
	for idx, val in enumerate(z):
		if abs(val) == lam2 and val * grad[idx] < 0:
			Gset.append((idx_old,idx+1))
			idx_old = idx+1
	if idx_old < len(v):
		Gset.append((idx_old, len(v)))

	for start, stop in Gset:
		if start == 0:
			tmp = - z[stop-1]
		elif stop == len(v):
			tmp = z[start-1]
		else:
			tmp = z[start-1] - z[stop-1]
		x[start:stop] = (sum(v[start:stop]) - tmp ) / (stop - start)

	return x 

def restart(z, v, grad, lam2):
	ux = v - w(z, v, grad, lam2)
	Rux = [ux[i]-ux[i-1] for i in xrange(1,len(v))]

	return Pro(Rose(Rux), lam2)

def GradDescent(v, z=None, lam2=1.0, maxiter=500, eps=1e-6):
	'''
	min 0.5 * ||R^Tz||^2 - v^T R^T z
	s.t. ||z||__ <= lam2

	grda: RR^T z - Rv 
	'''

	alpha = 0.25
	Rv = [v[i]-v[i-1] for i in xrange(1,len(v))] # R * v

	if z == None:
		z = uniform(-lam2, lam2, len(v))
	grad = RRTx(z) - Rv

	for i in xrange(maxiter):
		z = Pro(z - alpha * grad, lam2)
		grad = RRTx(z) - Rv

		# converage check 
		gap = lam2 * norm(grad, 1) + dot(z, grad)
		if abs(gap) < eps: break

		# restart
		z = restart(z, v, grad, lam2)
		grad = RRTx(z) - Rv

	return  w(z, v, grad, lam2)

def fusedlasso(v, lam2, lam1):
	u = [v[i]-v[i-1] for i in xrange(1,len(v))] # u = R * v
	z = Rose(u) # solve RR'z = Rv = u

	if max(abs(z)) <= lam2:
		# ||z||__ = lam2_max <= lam2
		x = v.mean() * ones(len(v))
	else:
		# ||z||__ = lam2_max > lam2
		x = GradDescent(v=v, z=Pro(z, lam2), lam2=lam2)

	res = abs(x) - lam1
	res[res < 0] = 0
	return sign(x) * res 

