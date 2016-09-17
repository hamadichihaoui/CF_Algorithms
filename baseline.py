
from __future__ import division

import math

import time
import operator

# read ratings file with entries (userid, movieid, rating)
def readratings(f):
    ratings = {}
    index=0
    somme=0
    for row in f:
        line = row.split("\t")        
        userid, movieid, rating = int(line[0]), int(line[1]), int(line[2])
        ratings.setdefault(userid, {})
        ratings[userid][movieid] = rating
	somme=somme+rating
	index=index+1
    

    return ratings,somme/index
def transpose(util):
    transposed = {}
    for id1 in util:
        for id2 in util[id1]:
            transposed.setdefault(id2, {})
            # flip id1 and id2
            transposed[id2][id1] = util[id1][id2]
    return transposed


def normalize(util):
    # save average of each user
    avgs = {}
    for id1 in util:
        avg = 0.0
        for id2 in util[id1]:
            avg += util[id1][id2]
        avg = float(avg)/len(util[id1])
        for id2 in util[id1]:
            util[id1][id2] -= avg
        avgs[id1] = avg
    return avgs

def bais_item(movies,mean,lamda=5):
		bais_items={}
		
		for item in movies:
			somme=0
			index=0
			for user in movies[item]:
			
				somme=somme+(movies[item][user]-mean)
				index=index+1
			bais_items[item]=somme/(index+lamda)
		return bais_items
def bais_user(users,mean,bais_items,lamda=5):
		bais_users={}
		
		for user in users:
			somme=0
			index=0
			for movie in users[user]:
			
				somme=somme+(users[user][movie]-mean-bais_items[movie])
				index=index+1
			bais_users[user]=somme/(index+lamda)
		return bais_users	




import operator
import numpy as np
def dcg_at_k(r, k, method=0):
	
	r = np.asfarray(r)[:k]
        if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			return np.sum(r / np.log2(np.arange(2, r.size + 2)))
		else:
			raise ValueError('method must be 0 or 1.')
	return 0
def ndcg_at_k(r, k, method=0):

        dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
		return 0.
	return dcg_at_k(r, k, method) / dcg_max

if __name__ == "__main__":
    init = time.time()
    # read in training data set
    f1 = open("ua.base")
    users,s = readratings(f1)
   
    f1.close()

    # read in test data set
    f2 = open("ua.test")
    rated,a = readratings(f2)
    f2.close()
    # normalize user ratings
    movies = transpose(users)
 
    b_items=bais_item(movies,s,lamda=5)
    b_users=bais_user(users,s,b_items,lamda=5)
    # computes similarities between all movies
 
    total = 0
    totalrmse = 0.0
    totalndcg=0
    l=[]
    for userid in rated:
	    list=[]
	    l1=0
	    for movieid in rated[userid]:
			    
		if movieid in movies:
				
			list.append((rated[userid][movieid],-(s+b_items[movieid]+b_users[userid])))
			totalrmse += (s+b_items[movieid]+b_users[userid]-rated[userid][movieid])**2
			l1=l1+(s+b_items[movieid]+b_users[userid]-rated[userid][movieid])**2
			total += 1
	    list.sort(key=operator.itemgetter(1))
	    totalndcg=totalndcg+ndcg_at_k([list[i][0] for i in xrange(len(list))],len(list))
	    l.append(l1/len(rated[userid]))	
	    #print total
	    #print totalrmse
    f3=open("a",'wb')
    for it in l:
	
	f3.write("%s\n" % it)
    print "user-based totalrmse: ", math.sqrt(totalrmse/total)
    print "ndcg=", totalndcg/len(rated)
    print time.time()-init
  