from splearn import lars, lars_path
from splearn import cwd_lasso, cwd_elasticnet
from splearn import plotpath, loadata
from numpy.random import uniform, randn
from numpy import dot 

X, y = loadata('dataset/regudata')

caltype = 'lar'
path = lars_path(X, y, method=caltype)
beta1 = lars(X, y, lam=100)
print beta1
beta2 = cwd_lasso(X, y, lam=100)
print beta2
beta3 = cwd_elasticnet(X, y, lam1=100, lam2=3.0)
print beta3


plotpath(path, pathtype=caltype.upper())

