import numpy as np
from numpy import mean
from operator import itemgetter
import numpy
from base import *
from pylab import *


def matrix_factorization(R,K,list,steps=30, alpha=0.01, beta=0.05):
    import random
    N=len(R)
    M=len(R[1])   
    P=numpy.random.uniform(0,0.1,(N,K))   
    Q =numpy.random.uniform(0,0.1,(M,K))
   
    Q = Q.T   
    list1=[i for i in xrange(K)]
    for step in xrange(steps):
	print step
	
	s=0
        #random.shuffle(list)
	for x in list:	
                x1=random.choice(list)
		i=x1[0]
		j=x1[1]
		eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
		s=s+eij**2		 
		for k in list1:			
                        P[i][k] = P[i][k] + alpha *(eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha *(eij * P[i][k] - beta * Q[k][j])
        s=math.sqrt(s/len(list))
                   	
    return P, Q.T

train = Data()
train.load('./ua.base', sep='	', format={'col':0, 'row':1, 'value':2, 'ids':int})
K=30
baseline_predictor=BASE()
baseline_predictor.set_data(train)
sparse_matrix=baseline_predictor.get_sparse_matrix()
print sparse_matrix
#bais_matrix=baseline_predictor.get_bais_matrix()
nP, nQ = matrix_factorization(sparse_matrix, K,baseline_predictor.list_couple_u_i)
nR = numpy.dot(nP, nQ.T)
users=baseline_predictor.get_users_without_occurence()
items=baseline_predictor.get_items_without_occurence()
#load test data
test=Data()
test.load('./ua.test', sep='	', format={'col':0, 'row':1, 'value':2, 'ids':int})
rmse = RMSE()
mae = MAE()

for rating, item_id, user_id in test.get():
    try:
	
	if (user_id in users and item_id in items):
		pred_rating = nR[users.index(user_id),items.index(item_id)]

		rmse.add(rating, pred_rating)
		mae.add(rating, pred_rating)
		
    except KeyError:
        continue
print 'RMSE=%s' % rmse.compute()
print 'MAE=%s' % mae.compute()

