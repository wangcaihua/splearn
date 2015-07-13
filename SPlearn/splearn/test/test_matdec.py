from numpy import dot, diag
from numpy.linalg import norm
from numpy.random import uniform
from splearn import snmf, ssvd, spcoding, spca 

n, m, k = 125, 25, 5
B0 = uniform(0,50,size=(n,k))
H0 = uniform(0,40,size=(k,m))
A = dot(B0,H0)

B, H = snmf(A, k, 0.1)
print 'OK!'



args1 = {'lam':0.7}
args2 = {'lam':0.5}
U, D, V = ssvd(A, 10, pfname1='sparseness', args1=args1, pfname2='sparseness', args2=args2)
re_A = dot(U.T, V)
re_A = dot(dot(U.T, diag(D)), V)
dif_A1 = A - re_A
print norm(A)
print norm(dif_A1)

U2, V2 = spca(A, 10, pfname='sparseness', args=args1)
print 'OK!'

base, code = spcoding(A, 10, pfname1='sparseness', args1=args1, pfname2='sparseness', args2=args2)
print code.shape 

