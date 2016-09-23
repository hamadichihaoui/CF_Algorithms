#To show some messages:

from numpy import mean
from operator import itemgetter
from base import *
import numpy
from pylab import *
import time


def matrix_factorization(R,K,mean,bais_items,bais_users,rated_items_for_each_user,list,steps=20,alpha1=0.01, alpha2=0.01,beta1=0.05,beta2=0.05):
    import random
    N=len(R)
    M=len(R[1])
  
   
    P=numpy.random.uniform(0,0.1,(N,K))
    Q =numpy.random.uniform(0,0.1,(M,K))
    Y=numpy.random.uniform(0,0.1,(M,K))
   
    Q = Q.T
    
  
    
    for step in xrange(steps):
	print step
	s=0
        random.shuffle(list)
	#nn=10000
	for x in list :	
		i=x[0]
		j=x[1]
		somme=np.zeros((1,K))
		for  item in rated_items_for_each_user[i]:
			somme[0]=somme[0]+Y[item,:]
		somme[0]=somme[0]*(1/math.sqrt(len( rated_items_for_each_user[i])))
		eij = R[i][j] - mean-bais_items[j]-bais_users[i]-numpy.dot(P[i,:],Q[:,j])-numpy.dot(somme[0],Q[:,j])
		s=s+eij**2
		   	
		bais_items[j]=bais_items[j]+alpha1*(eij-beta1* bais_items[j])
		
		bais_users[i]=bais_users[i]+alpha1*(eij-beta1 * bais_users[i])
		for k in xrange(K):
			
			P[i][k] = P[i][k] + alpha2 * (eij * Q[k][j] - beta2 * P[i][k])
                        Q[k][j] = Q[k][j] + alpha2 * (eij * (P[i][k]+somme[0][k]) - beta2 * Q[k][j])
		for  item in rated_items_for_each_user[i]:
			Y[item,:]=Y[item,:]+alpha2*(eij/math.sqrt(len( rated_items_for_each_user[i]))* Q[:,j] - beta2 * Y[item,:])
                   
	print math.sqrt(s/len(list))		
    return P, Q.T,bais_items,bais_users,Y



init=time.time()
train = Data()
train.load('./ua.base', sep='	', format={'col':0, 'row':1, 'value':2, 'ids':int})
K=40


baseline_predictor=BASE()
baseline_predictor.set_data(train)
sparse_matrix=baseline_predictor.get_sparse_matrix()
print sparse_matrix
print len(sparse_matrix)
print len(sparse_matrix[1])
mean=baseline_predictor.get_mean_ratings()
users=baseline_predictor.get_users_without_occurence()
items=baseline_predictor.get_items_without_occurence()
rated_items_for_each_user=[]
for user in xrange(len(users)):
	rated_items_for_each_user.append(baseline_predictor.support_user(user,user))
bais_items=baseline_predictor.bais_items
print len(bais_items)
bais_users=baseline_predictor.bais_users
print len(bais_users)



#bais_matrix=baseline_predictor.get_bais_matrix()
nP, nQ,b_i,b_u,Y = matrix_factorization(sparse_matrix, K,mean,bais_items,bais_users,rated_items_for_each_user,baseline_predictor.list_couple_u_i)
nR = numpy.dot(nP, nQ.T)

Y1=np.zeros(((len(users)),K))
for user in xrange(len(users)):
	for  item in rated_items_for_each_user[user]:
		Y1[user,:]=Y1[user,:]+Y[item,:]
	Y1[user,:]=Y1[user,:]*(1/math.sqrt(len( rated_items_for_each_user[user])))
nR=nR+numpy.dot(Y1, nQ.T)



#load test data
test=Data()
test.load('./ua.test', sep='	', format={'col':0, 'row':1, 'value':2, 'ids':int})
rmse = RMSE()
mae = MAE()

for rating, item_id, user_id in test.get():
    try:
	#print rating
	if (user_id in users and item_id in items):
		pred_rating = nR[users.index(user_id),items.index(item_id)]+b_i[items.index(item_id)]+b_u[users.index(user_id)]+mean
	#print rating
		#print pred_rating,rating
		rmse.add(rating, pred_rating)
		mae.add(rating, pred_rating)
		
    except KeyError:
        continue
print 'RMSE=%s' % rmse.compute()
print 'MAE=%s' % mae.compute()

print time.time()-init
