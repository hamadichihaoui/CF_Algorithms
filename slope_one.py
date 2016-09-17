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
    
    
    
def cosine_sim(util, id1, id2):
	    num = 0
		
	    # get items util[id1] and util[id2] share in common
	    shared = set(util[id1].keys()).intersection(util[id2].keys())
	   
	    s=0
	    for user in shared:
		s=s+util[id1][user]-util[id2][user]
	    x = s/len(shared) if len(shared) != 0 else 0
	    return(x,len(shared))
    
    
    
def computesims(util):
	    sims = {}
	    for id1 in util:
		sims[id1] = {}
		for id2 in util:
		    if id1 == id2: continue
	
		    sims[id1][id2] = cosine_sim(util, id1, id2)
            return sims
    
    
def itembased(users, mtopredict, moviessim, id1):
    items = {}
    simsums = {}
    predictions = {}

    for item in mtopredict:
        items[item] = 0.0
        simsums[item] = 0.0


    for item in items:
        # initialize heap for item
      
        for item2 in users[id1]:
            if item == item2: continue
            # some movies like the one with id 1582
            # hasn't been rated by any user in ua.base
            if item not in moviessim or item2 not in moviessim[item]:
                predictions[item] = 0.0
                continue

            (sim, lenshared) = moviessim[item][item2]
	    items[item] += (users[id1][item2]+sim)*lenshared
            simsums[item] += lenshared
           
           
        
           
    for item in items:
        predictions[item] = items[item] / simsums[item] if simsums[item] > 0 else 0

    return predictions
    
    
    
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
    
 
    #for user in b_users:
	   #print user,b_users[user]
    
    #mpredictions = {}
    init = time.time()
   
    # computes similarities between all movies
    
    moviessim = computesims(movies)
    
    list_neigbors1=[60]
    #for movie in moviessim:
	    #for x in moviessim[movie]:
		    #print moviessim[movie][x]
    for n1 in  list_neigbors1:
	 
	    totalndcg=0
	    totalrmse = 0.0
	    total = 0
	    for userid in rated:
		#list=[]
		predictions = itembased(users, rated[userid].keys(), moviessim, userid)
		for movieid in rated[userid]:
		    if movieid in predictions and movieid in movies:
			totalrmse += (predictions[movieid]-rated[userid][movieid])**2
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