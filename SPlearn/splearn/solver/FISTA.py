# Author: Caihua Wang <490419716@qq.com>

from math import sqrt
from numpy import dot
from numpy.linalg import norm
from numpy.random import uniform


__all__ = ['fista']


def fista(objfunc, dfunc, p, Profunc, regu, args={'lam':100.0}, theta=None, eta=2, L0=1.0, maxiter=500, eps=1e-6):
	if theta == None:
		theta = uniform(-0.1, 0.1, size=p)
	x_old, y = theta, theta
	tk_old, L, i = 1.0, L0, 0

	while i < maxiter:
		fy, dfy = objfunc(y), dfunc(y)

		# step one
		while True:
			xk = Profunc(y, dfy, L, regu, args)
			if objfunc(xk) <= fy + dot(xk-y, dfy) + (L/2.0) * norm(xk-y)**2:
				break
			else:
				L *= eta

		# step two
		tk = (1.0 + sqrt(1.0+4.0*tk_old**2))/2.0
		dx = xk - x_old
		y = xk + (tk_old-1.0)/tk * dx
		tk_old = tk
		x_old = xk
		i += 1

		if norm(dx) <= eps:	
			return xk

	return xk 
