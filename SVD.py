#To show some messages:

from numpy import mean
from operator import itemgetter
from base import *
import numpy
from pylab import *


def matrix_factorization(R,K,mean,bais_items,bais_users,list,steps=15, alpha=0.01,beta=0.05):
    import random
    N=len(R)
    M=len(R[1])
  
   
    P=numpy.random.uniform(0,0.1,(N,K))
    Q =numpy.random.uniform(0,0.1,(M,K))
    
    Q = Q.T
    
    olderr=1000
    list1=[i for i in xrange(K)]
    olderr=100
    newerr=0
    for step in xrange(steps):
	print step
	
	s=0
        random.shuffle(list)	
	for x in list:	
                
		i=x[0]
		j=x[1]
		eij = R[i][j] - mean-bais_items[j]-bais_users[i]-numpy.dot(P[i,:],Q[:,j])
		s=s+eij**2
		bais_items[j]=bais_items[j]+alpha*(eij-beta * (bais_items[j]))
		
		bais_users[i]=bais_users[i]+alpha*(eij-beta * (bais_users[i]))
		for k in xrange(K):
			
			P[i][k] = P[i][k] + alpha * (eij * Q[k][j] - beta * (P[i][k]))
                        Q[k][j] = Q[k][j] + alpha * (eij * P[i][k] - beta * (Q[k][j]))
        olderr=newerr
	newerr=math.sqrt(s/len(list))
	print newerr
	return P, Q.T,bais_items,bais_users




train = Data()
train.load('./ua.base', sep='	', format={'col':0, 'row':1, 'value':2, 'ids':int})
K=70


baseline_predictor=BASE()
baseline_predictor.set_data(train)
sparse_matrix=baseline_predictor.get_sparse_matrix()
print sparse_matrix
mean=baseline_predictor.get_mean_ratings()
bais_items=baseline_predictor.bais_items
print len(bais_items)
bais_users=baseline_predictor.bais_users
print len(bais_users)
nP, nQ,b_i,b_u = matrix_factorization(sparse_matrix, K,mean,bais_items,bais_users,baseline_predictor.list_couple_u_i)
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
		pred_rating = nR[users.index(user_id),items.index(item_id)]+b_i[items.index(item_id)]+b_u[users.index(user_id)]+mean
	
		rmse.add(rating, pred_rating)
		mae.add(rating, pred_rating)
		
    except KeyError:
        continue
print 'RMSE=%s' % rmse.compute()
print 'MAE=%s' % mae.compute()

