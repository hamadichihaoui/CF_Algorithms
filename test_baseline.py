from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import operator
import math
import heapq
import time

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
def cosine_sim(util, id1, id2,mean,b_i,b_u, th=2):
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

    return (num*len(shared)/(denom*(80+len(shared))), len(shared))

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
            sims[id1][id2] = cosine_sim(util, id1, id2,s,b_i,b_u)
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

def itembased(users, mtopredict, moviessim, id1,s,b_i,b_u, n=50,th=1):
    items = {}
    simsums = {}
    predictions = {}

    for item in mtopredict:
        items[item] = 0.0
        simsums[item] = 0.0


    for item in items:
      
        h = []
        for item2 in users[id1]:
            if item == item2: continue
           
            if item not in moviessim or item2 not in moviessim[item]:
                predictions[item] = 0.0
                continue

            (sim, lenshared) = moviessim[item][item2]
            if lenshared < th: continue
            heapq.heappush(h, (-sim, item2))
      
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
    f1.close()
    # read in test data set
    f2 = open("ua.test")
    rated,a = readratings(f2)    
    movies = transpose(users)

    b_items=bais_item(movies,s,lamda=5)
    b_users=bais_user(users,s,b_items,lamda=5)
    mpredictions = {}     
    # computes similarities between all movies    
    moviessim = computesims(movies,s,b_items,b_users)
    totalndcg=0
    totalrmse = 0.0
    total = 0
    for userid in rated:
		list=[]
		predictions = itembased(users, rated[userid].keys(), moviessim, userid,s,b_items,b_users)
		for movieid in rated[userid]:
		    if movieid in predictions and movieid in movies:
			totalrmse += (predictions[movieid]+s+b_items[movieid]+b_users[userid]-rated[userid][movieid])**2
			list.append((rated[userid][movieid],-(s+b_items[movieid]+b_users[userid]+predictions[movieid])))
			total += 1
		list.sort(key=operator.itemgetter(1))
		totalndcg=totalndcg+ndcg_at_k([list[i][0] for i in xrange(len(list))],len(list))
		
	   
    print "RMSE= ", math.sqrt(totalrmse/total)
    print "NDCG=",totalndcg/len(rated)
    print "elapsed time=", time.time()-init
  
    
    
   

    
    
    