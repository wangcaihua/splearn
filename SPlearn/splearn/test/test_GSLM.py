from numpy.linalg import norm 
from splearn import loadata, accmse, GSLM

GroupedLoss = [ 'squarederror', 	  #regression
				'huberloss', 		  #regression
				'logit', 			  #classification
				'modifiedhuberloss',  #classification
				'huberizedhinge', 	  #classification
				'squaredhinge'		  #classification
				]

GroupedRegu = [ 'grouplasso', 
				'sparsegrouplasso']


##########################  Parameters #################################
loss = 'squarederror'
delta = 0.9
grouplist = [(0,3), (3,5), (5, 8), (8,10)]
groupweight = None
regu = 'grouplasso'
reargs = {'lam':0.5}
########################################################################
if loss in ['squarederror', 'huberloss']:
	fname = 'dataset/regudata'
	calmod = 'mse'
else:
	fname = 'dataset/clasdata'
	calmod = 'acc'
########################################################################

# load data set
data, target = loadata(fname)
# learning and predict ...
#args --> (loss, delta, regu, reargs, grouplist, groupweight)
LS = GSLM(loss=loss, delta=delta, regu=regu, reargs=reargs, grouplist=grouplist, groupweight=groupweight)
LS.fit(data, target)
pred = LS.predict(data)
res = accmse(target, pred, calmod=calmod)


print LS.coeff_
print res 
