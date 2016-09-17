
from __future__ import division
import matplotlib.pyplot as plt
import math
import heapq
import time

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
def cosine_sim(util, id1, id2,mean,b_i,b_u, th=1):
    num = 0
        
    # get items util[id1] and util[id2] share in common
    shared = set(util[id1].keys()).intersection(util[id2].keys())

    # optimization to not compute similarity between items
    # that don't meet threshold
    if len(shared) < th:
        return (0.0, len(shared))
    firstmag = 0 
    secondmag = 0
    # calculate dot product and magnitudes of shared items
    for item in shared:
	x=util[id1][item]-mean-b_i[id1]-b_u[item]
	y=util[id2][item]-mean-b_i[id2]-b_u[item]
        num +=x * y
        firstmag += x**2
        secondmag += y**2

    # prevent denom == 0
    firstmag = 1 if firstmag == 0 else firstmag
    secondmag = 1 if secondmag == 0 else secondmag
    # calculate magnitude of shared items in util[id2]
    denom = math.sqrt(firstmag) * math.sqrt(secondmag)

    return (num*len(shared)/(denom*(60+len(shared))), len(shared))

def cosine_sim1(util, id1, id2,mean,b_i,b_u, th=1):
    num = 0
        
    # get items util[id1] and util[id2] share in common
    shared = set(util[id1].keys()).intersection(util[id2].keys())

    # optimization to not compute similarity between items
    # that don't meet threshold
    if len(shared) < th:
        return (0.0, len(shared))
    firstmag = 0 
    secondmag = 0
    # calculate dot product and magnitudes of shared items
    for item in shared:
        num += (util[id1][item]-mean-b_i[id1]-b_u[item] )* (util[id2][item]-mean-b_i[id2]-b_u[item])
    for item in util[id1]:
        firstmag += (util[id1][item]-mean-b_i[id1]-b_u[item] )**2
    for item in util[id2]:
        secondmag += (util[id2][item]-mean-b_i[id2]-b_u[item])**2

    # prevent denom == 0
    firstmag = 1 if firstmag == 0 else firstmag
    secondmag = 1 if secondmag == 0 else secondmag
    # calculate magnitude of shared items in util[id2]
    denom = math.sqrt(firstmag) * math.sqrt(secondmag)

    return (num*len(shared)/(denom*(len(shared)+20)), len(shared))

def computesims(util,s,b_i,b_u):
    sims = {}
    for id1 in util:
        sims[id1] = {}
        for id2 in util:
            if id1 == id2: continue
            sims[id1][id2] = cosine_sim1(util, id1, id2,s,b_i,b_u)
    return sims
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

def itembased(users, mtopredict, moviessim, id1,s,b_i,b_u, n,th=1):
    items = {}
    simsums = {}
    predictions = {}

    for item in mtopredict:
        items[item] = 0.0
        simsums[item] = 0.0


    for item in items:
        # initialize heap for item
        h = []
        for item2 in users[id1]:
            if item == item2: continue
            # some movies like the one with id 1582
            # hasn't been rated by any user in ua.base
            if item not in moviessim or item2 not in moviessim[item]:
                predictions[item] = 0.0
                continue

            (sim, lenshared) = moviessim[item][item2]
            if lenshared < th: continue
            heapq.heappush(h, (-sim, item2))
        # get the nearest neightbors for item
        nn = 0
        while nn < n and len(h) > 0:
            (sim, item2) = heapq.heappop(h)
            sim = -sim
            items[item] += (users[id1][item2]-s-b_i[item2]-b_u[id1])*sim
            simsums[item] += abs(sim)
            nn += 1
    for item in items:
        predictions[item] = items[item] / simsums[item] if simsums[item] > 0 else 0

    return predictions

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
    init=time.time()
    # read in training data set
    f1 = open("ua.base")
    users,s = readratings(f1)
    print s
    #print users
    f1.close()

    # read in test data set
    f2 = open("ua.test")
    rated,a = readratings(f2)
    
    # normalize user ratings
   
    movies = transpose(users)
    #for movie in movies:
	    #print movie
	    #for user in movies[movie]:
		    #print user,movies[movie][user]
    #avgs = normalize(movies)
    
    b_items=bais_item(movies,s,lamda=5)
    #for item in b_items:
	    #print item,b_items[item]
    b_users=bais_user(users,s,b_items,lamda=5)
    #for user in b_users:
	   #print user,b_users[user]
    
    mpredictions = {}
    init = time.time()
   
    # computes similarities between all movies
    
    moviessim = computesims(movies,s,b_items,b_users)
    l=[1600]
    list_neigbors=[5,10,15,20,25,30,35,40,45,50,60,70,80190,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,1000,1200,1400,1600,1680]
    list_rmse=[]
    list_ndcg=[]
    list_neigbors1=[60]
    #for movie in moviessim:
	    #for x in moviessim[movie]:
		    #print moviessim[movie][x]
    for n1 in  list_neigbors1:
	    print n1
	    totalndcg=0
	    totalrmse = 0.0
	    total = 0
	    for userid in rated:
		#list=[]
		predictions = itembased(users, rated[userid].keys(), moviessim, userid,s,b_items,b_users,n=n1)
		for movieid in rated[userid]:
		    if movieid in predictions and movieid in movies:
			totalrmse += (predictions[movieid]+s+b_items[movieid]+b_users[userid]-rated[userid][movieid])**2
			#list.append((rated[userid][movieid],-(s+b_items[movieid]+b_users[userid])))
			#print avgs[movieid],rated[userid][movieid]
			#mpredictions.setdefault(movieid, (movieid, 0.0, 0))
			#movieid, crmse, nest = mpredictions[movieid]
			#mpredictions[movieid] =  (movieid, crmse+(predictions[movieid]+avgs[userid]-rated[userid][movieid])**2, nest+1)
			total += 1
		#list.sort(key=operator.itemgetter(1))
		#print [list[i][0] for i in xrange(len(list))]
		#totalndcg=totalndcg+ndcg_at_k([list[i][0] for i in xrange(len(list))],len(list))
		
	    #print total
	    print "item-based totalrmse: ", math.sqrt(totalrmse/total)
	    #print "ndcg=",totalndcg/len(rated)
	    #print len(rated)
	    #print "time taken: ", time.time()-init
	    list_rmse.append( math.sqrt(totalrmse/total))
	    #list_ndcg.append(totalndcg/len(rated))
	    #print "time taken: ", time.time()-init
  
    
    
    '''plt.plot( list_neigbors1, list_ndcg, 'bo')
    #plt.plot( list_neighbors, list_rmse, 'k')
    plt.axis([0,1700, 0.94,0.96])
    plt.xlabel('number of nearest neighbors')
    plt.ylabel('NDCG')
    plt.show()
    plt.plot(list_neigbors, list_rmse, 'ro')
    #plt.plot( list_neighbors, list_rmse, 'k')
    plt.axis([0,1700, 0.93,0.99])
    plt.xlabel('number of nearest neighbors')
    plt.ylabel('RMSE')
    plt.show()'''

    
    
    