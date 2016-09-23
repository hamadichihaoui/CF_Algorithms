from __future__ import division
import numpy as np
import operator
import math
import time
import operator

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
            transposed[id2][id1] = util[id1][id2]
    return transposed

def normalize(util):
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


def cosine_sim(util, id1, id2,mean,b_i,b_u, th=1):
    num = 0
    shared = set(util[id1].keys()).intersection(util[id2].keys())
    if len(shared) < th:
        return (0.0, len(shared))
    firstmag = 0 
    secondmag = 0
    for item in shared:
        x=util[id1][item]-mean-b_i[item]-b_u[id1]
        y=util[id2][item]-mean-b_i[item]-b_u[id2]
        num +=x * y
        firstmag += x**2
        secondmag += y**2
    firstmag = 1 if firstmag == 0 else firstmag
    secondmag = 1 if secondmag == 0 else secondmag
    denom = math.sqrt(firstmag) * math.sqrt(secondmag)
    return (num*len(shared)/(denom*(len(shared)+80)), len(shared))
    
def pearson(util, id1, id2,mean,b_i,b_u, th=1):
    num = 0        
    shared = set(util[id1].keys()).intersection(util[id2].keys())  
    if len(shared) < th:
        return (0.0, len(shared))
    firstmag = 0 
    secondmag = 0
    
    for item in shared:
        x=util[id1][item]-mean-b_i[item]-b_u[id1]
        y=util[id2][item]-mean-b_i[item]-b_u[id2]
        num +=x * y
    for item in util[id1]:
        firstmag += (util[id1][item]-mean-b_i[item]-b_u[id1])**2
    for item in util[id2]:
        secondmag += (util[id2][item]-mean-b_i[item]-b_u[id2])**2
    firstmag = 1 if firstmag == 0 else firstmag
    secondmag = 1 if secondmag == 0 else secondmag
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
		return 0
	return dcg_at_k(r, k, method) / dcg_max
	
def userbased(users,movies, mtopredict, userssim, id1,s,b_i,b_u,n,th=1):
    predictions = {}
    heaps = {}
    items = {}
    simsums = {}
    nn = 0
    for item in mtopredict:
        items[item] = 0.0
        simsums[item] = 0.0
    for item in mtopredict:
	h1=[]
	if item in movies:
		
		for id2 in movies[item] :
			(sim, lenshared) = userssim[id1][id2]			
			if lenshared < th: continue
			else:				
				h1.append((-sim,id2))		
		h1.sort(key=operator.itemgetter(0))
		nn = 0
		while len(h1)> nn and nn<n:
			(sim,id2)=h1[nn]
			
			sim = -sim
			items[item] += (users[id2][item]-s-b_i[item]-b_u[id2])*sim
			simsums[item] += abs(sim)
			nn += 1
		predictions[item] = items[item] / simsums[item] if simsums[item] > 0 else 0
    return predictions

if __name__ == "__main__":
	
    init = time.time()
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
    # computes similarities between all movies
    usersim = computesims(users,s,b_items,b_users)
   
    total = 0
    totalrmse = 0.0
    totalndcg=0
    for userid in rated:
	    list=[]
	    predictions = userbased(users,movies, rated[userid].keys(), usersim, userid,s,b_items,b_users,n=50,th=1)
		
	    for movieid in rated[userid]:
			    
		if movieid in predictions and movieid in movies:
				
			list.append((rated[userid][movieid],-(predictions[movieid]+s+b_items[movieid]+b_users[userid])))
			totalrmse += (predictions[movieid]+s+b_items[movieid]+b_users[userid]-rated[userid][movieid])**2
			total += 1
	    list.sort(key=operator.itemgetter(1))
	    totalndcg=totalndcg+ndcg_at_k([list[i][0] for i in xrange(len(list))],len(list))
	
    print "RMSE= ", math.sqrt(totalrmse/total)
    print "NDCG=", totalndcg/len(rated)
    print "time elapsed",time.time()-init
  