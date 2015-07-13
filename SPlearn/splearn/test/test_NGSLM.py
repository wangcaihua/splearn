from splearn import loadata, accmse, NGSLM

NonGroupedLoss = ['squarederror', 	#regression
				  'huberloss', 		#regression
				  'logit', 			#classification
				  'modifiedhuberloss', 	#classification
				  'huberizedhinge', 	#classification
				  'squaredhinge'		#classification
				  ]

NonGroupedRegu = ['lasso', 
				  'elasticnet', 
				  'fusedlasso']


##########################  Parameters #################################
loss = 'huberloss'
delta = 10

regu = 'lasso'
reargs = {'lam':30}
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
LS = NGSLM(loss=loss, delta=delta, regu=regu, reargs=reargs)
LS.fit(data, target)
pred = LS.predict(data)
res = accmse(target, pred, calmod=calmod)

print LS.coeff_
print res 

