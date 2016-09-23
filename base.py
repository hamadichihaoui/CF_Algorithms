import sys
VERBOSE = False #Set to True to get some messages
#To show some messages:
from math import sqrt
from scipy.stats import pearsonr

#from recsys.evaluation import ROUND_FLOAT
ROUND_FLOAT=6
import numpy as np


from operator import itemgetter
import numpy
from pylab import *


class BASE(object):
	def __init__(self):
		self.data = Data()
		self.matrix = None
		self.bais_matrix=None
		self.list_couple_u_i=[]
		self.mean_ratings=0
		self.bais_items=[]
		self.bais_users=[]
	def set_data(self,data):
		self.data=data
	def get_sparse_matrix(self):
		users1=self.get_users_without_occurence()
		users2=self.get_users_with_occurence()
		items1=self.get_items_without_occurence()
		items2=self.get_items_with_occurence()
		self.matrix=zeros((len(users1),len(items1)))
		ratings=map(itemgetter(0),self.data)
		
		index1=0
		somme=0
		for rating in ratings :#i parcoure ratings
			j1=users1.index(users2[index1])
			j2=items1.index(items2[index1])
			self.matrix[j1,j2]=rating
			self.list_couple_u_i.append((j1,j2))
			somme=somme+rating
			index1=index1+1
		self.mean_ratings=somme/index1
		self.bais_items=[]
		self.bais_users=[]
		#mean_rating=self.mean_ratings()
		#items=items_without_occurence(data)
		#users=users_without_occurence(data)
		for i in xrange(0,len(self.matrix[1])):
			self.bais_items.append(self.bais_item(i))
		for u in xrange(0,len(self.matrix)):
			self.bais_users.append(self.bais_user(u))
		return self.matrix

	def support_user(self,u1,u2):
		list=[]
		for  j in range (0, len(self.matrix[1])):
			if self.matrix[u1,j]*self.matrix[u2,j]!=0:
				list.append(j)
		return list

	def get_bais_matrix(self,lamda1,lamda2):
		self.bais_items=[]
		self.bais_users=[]
		#mean_rating=self.mean_ratings()
		#items=items_without_occurence(data)
		#users=users_without_occurence(data)
		for i in xrange(0,len(self.matrix[1])):
			self.bais_items.append(self.bais_item(i,lamda1))
		for u in xrange(0,len(self.matrix)):
			self.bais_users.append(self.bais_user(u,lamda2))
		
		self.bais_matrix=zeros((len(self.bais_users),len(self.bais_items)))
		for i1 in range(0,len(self.bais_users)):
			for j1 in range (0,len(self.bais_items)):
				self.bais_matrix[i1,j1]=self.mean_ratings+self.bais_users[i1]+self.bais_items[j1]	
		
		return self.bais_matrix
		
	def get_mean_ratings(self):
		return mean(map(itemgetter(0),self.data))
	def get_items_with_occurence(self):
		return  map(itemgetter(1),self.data)
	def get_items_without_occurence(self):
		items=[]
		list=[]
		list=self.get_items_with_occurence()
		for i in list:
			if not( i in items):
				items.append(i)
		return items
		

	def get_users_with_occurence(self):
		return  map(itemgetter(2),self.data)
	def get_users_without_occurence(self):
		users=[]
		list=[]
		list=self.get_users_with_occurence()
		for i in list:
			if not( i in users):
				users.append(i)
		return users
		

	  
	def bais_item(self,j,lamda=5):
		somme=0
		index=0
		for i in xrange (0,len(self.matrix)):
			if self.matrix[i,j]!=0:
				somme=somme+self.matrix[i,j]-self.mean_ratings
				index=index+1
		return somme/(index+lamda) 
	def bais_user(self,i,lamda=5):
		somme=0
		index=0
		for j in xrange(0,len(self.matrix[1])):
			if self.matrix[i,j]!=0:
				somme=somme+self.matrix[i,j]-self.mean_ratings-self.bais_items[j]
				index=index+1
		return somme/(index+lamda)

class item_neighberhood(BASE):
	def __init__(self):
        #Call parent constructor
		super(item_neighberhood, self).__init__()
		self.similarity=None
		self.matrix=None
		
	def get_sparse_matrix(self):
		
		users1=self.get_users_without_occurence()
		users2=self.get_users_with_occurence()
		items1=self.get_items_without_occurence()
		items2=self.get_items_with_occurence()
		self.matrix=zeros((len(users1),len(items1)))
		self.similarity=zeros((len(self.matrix[1]),len(self.matrix[1])))
		ratings=map(itemgetter(0),self.data)
		
		index1=0
		somme=0
		for rating in ratings :#i parcoure ratings
			j1=users1.index(users2[index1])
			j2=items1.index(items2[index1])
			self.matrix[j1,j2]=rating
			somme=somme+rating
			index1=index1+1
		self.mean_ratings=somme/index1
		return self.matrix

	#determine les 
	def support_item(self,i1,i2):
		list=[]
		for  u in range (0, len(self.matrix)):
			if self.matrix[u,i1]*self.matrix[u,i2]!=0:
				list.append(u)
		return list
	def support_user(self,u1,u2):
		list=[]
		for  j in range (0, len(self.matrix[1])):
			if self.matrix[u1,j]*self.matrix[u2,j]!=0:
				list.append(j)
		return list

	def similarity_ha(self,i1,i2):
		list=[]
		result=0
		list=support_item(i1,i2)
		if list:
			vect1 = numpy.zeros(shape=(len(list)))
			vect2 = numpy.zeros(shape=(len(list)))
			i=0
			for j in list:
				vect1[i]=self.matrix[j,i1]
				vect2[i]=self.matrix[j,i2]
				i=i+1
			result =self.pearson_cor(vect1,vect2)
		return result
	def similarityU_ha(self,u1,u2):
		list=[]
		result=0
		list=support_user(u1,u2)
		if list:
			vect1 = numpy.zeros(shape=(len(list)))
			vect2 = numpy.zeros(shape=(len(list)))
			i=0
			for j in list:
				vect1[i]=self.matrix[j,i1]
				vect2[i]=self.matrix[j,i2]
				i=i+1
			result = pearson_cor(vect1,vect2)
		return result
	def Similarity_matrix(self):
		sim_mat=zeros((len(self.matrix[1]),len(self.matrix[1])))
		for i in range(0,len(sim_mat)):
			for j in range(0,len(sim_mat)):
				if i>=j:
					try:
						sim_mat[i,j]=similarity_ha(sparsematrice,i,j)
						sim_mat[j,i]=sim_mat[i,j]
					except:
						pass
		return sim_mat
	
	def pearson_cor(self,i,j):
		result=0
		common_users=[]
		common_users=self.support_item(i,j)
		if common_users:
			pij=0
			s1=0
			s2=0
			mui=0
			muj=0
			for u in common_users:
				mui=self.matrix[u,i]-self.bais_matrix[u,i]
				muj=self.matrix[u,j]-self.bais_matrix[u,j]
				pij=pij+mui*muj
				s1=s1+math.pow(mui,2)
				s2=s2+math.pow(muj,2)
		
		
			result=pij/math.sqrt(s1*s2)
		return result

	def predict(self,u,j):
		
		already_rated_items=[]
		
		already_rated_items=self.support_user(u,u)
		#corela_list=[]
		fuj=0
		s=0
		for i in already_rated_items:
			if self.similarity[i,j]==0:
				self.similarity[i,j]=self.pearson_cor(j,i)
				self.similarity[j,i]=self.similarity[i,j]
				fuj=fuj+self.similarity[i,j]*(self.matrix[u,i]-self.bais_matrix[u,i])
				s=s+abs(self.similarity[i,j])
				#corela_list.append(self.similarity[i,j])
			else:
				#corela_list.append(self.similarity[i,j])
				fuj=fuj+self.similarity[i,j]*(self.matrix[u,i]-self.bais_matrix[u,i])
				s=s+abs(self.similarity[i,j])
				
				
		#s=0
		#for i in corela_list:
		#	s=s+abs(i)
			
		#fuj=0
		#n=0
		#for i in corela_list:
		#	j1=already_rated_items[n]
		#	fuj=fuj+i*(self.matrix[u,j1]-self.bais_matrix[u,j1])
		#	n=n+1
		return self.bais_matrix[u,j]+ fuj/s
		
class user_neighberhood(BASE):
	def __init__(self, filename=None):
        #Call parent constructor
		super(user_neighberhood, self).__init__()
		self.similarity=None
	def get_sparse_matrix(self):
		users1=self.get_users_without_occurence()
		users2=self.get_users_with_occurence()
		items1=self.get_items_without_occurence()
		items2=self.get_items_with_occurence()
		self.matrix=zeros((len(users1),len(items1)))
		self.similarity=zeros((len(self.matrix),len(self.matrix)))
		ratings=map(itemgetter(0),self.data)
		
		index1=0
		somme=0
		for rating in ratings :#i parcoure ratings
			j1=users1.index(users2[index1])
			j2=items1.index(items2[index1])
			self.matrix[j1,j2]=rating
			somme=somme+rating
			index1=index1+1
		self.mean_ratings=somme/index1
		return self.matrix

		
		




  

#determine les 
	def support_item(self,i1,i2):
		list=[]
		for  u in range (0, len(self.matrix)):
			if self.matrix[u,i1]*self.matrix[u,i2]!=0:
				list.append(u)
		return list
	def support_user(self,u1,u2):
		list=[]
		for  j in range (0, len(self.matrix[1])):
			if self.matrix[u1,j]*self.matrix[u2,j]!=0:
				list.append(j)
		return list

	def similarity_item(self,i1,i2):
		list=[]
		result=0
		list=support_item(i1,i2)
		if list:
			vect1 = numpy.zeros(shape=(len(list)))
			vect2 = numpy.zeros(shape=(len(list)))
			i=0
			for j in list:
				vect1[i]=self.matrix[j,i1]
				vect2[i]=self.matrix[j,i2]
				i=i+1
			result =self.pearson_cor(vect1,vect2)
		return result
	'''def similarity_user(self,u1,u2):
		list=[]
		result=0
		list=support_user(u1,u2)
		if list:
			vect1 = numpy.zeros(shape=(len(list)))
			vect2 = numpy.zeros(shape=(len(list)))
			i=0
			for j in list:
				vect1[i]=self.matrix[j,i1]
				vect2[i]=self.matrix[j,i2]
				i=i+1
			result = self.pearson_cor(vect1,vect2,is_item=False))
		return result
	def Similarity_item_matrix(self):
		sim_mat=zeros((len(self.matrix[1]),len(self.matrix[1])))
		for i in range(0,len(sim_mat)):
			for j in range(0,len(sim_mat)):
				if i>=j:
					try:
						sim_mat[i,j]=similarity_item(sparsematrice,i,j)
						sim_mat[j,i]=sim_mat[i,j]
					except:
						pass
		return sim_mat'''
	
	def pearson_cor(self,i,j,is_item=True):
		if is_item:
			result=0
			common_users=[]
			common_users=self.support_item(i,j)
			if common_users:
				pij=0
				s1=0
				s2=0
				mui=0
				muj=0
				for u in common_users:
					mui=self.matrix[u,i]-self.bais_matrix[u,i]
					muj=self.matrix[u,j]-self.bais_matrix[u,j]
					pij=pij+mui*muj
					s1=s1+mui**2
					s2=s2+muj**2
			
			
				result=pij/math.sqrt(s1*s2)
		else:
			result=0
			common_items=[]
			common_items=self.support_item(i,j)
			if common_items:
				pij=0
				s1=0
				s2=0
				mui=0
				muj=0
				for it in common_items:
					mui=self.matrix[i,it]-self.bais_matrix[i,it]
					muj=self.matrix[j,it]-self.bais_matrix[j,it]
					pij=pij+mui*muj
					s1=s1+mui**2
					s2=s2+muj**2
			
			
				result=(len(common_items)-1)*pij/(math.sqrt(s1*s2)*(len(common_items)-1+100))
			
		return result

	def predict(self,u,j):
		
		users_that_rated_j=[]
		
		users_that_rated_j=self.support_item(j,j)
		#corela_list=[]
		fuj=0
		s=0
		for v in users_that_rated_j:
			#print v
			if self.similarity[u,v]==0:
				self.similarity[u,v]=self.pearson_cor(u,v,is_item=False)
				self.similarity[v,u]=self.similarity[u,v]
				#if (self.similarity[u,v]>0.1):
				fuj=fuj+self.similarity[u,v]*(self.matrix[v,j]-self.bais_matrix[v,j])
				s=s+abs(self.similarity[u,v])
				#corela_list.append(self.similarity[i,j])
			else:
				#if (self.similarity[u,v]>0.1):
				#corela_list.append(self.similarity[i,j])
				fuj=fuj+self.similarity[u,v]*(self.matrix[v,j]-self.bais_matrix[v,j])
				s=s+abs(self.similarity[u,v])
					
				
		#s=0
		#for i in corela_list:
		#	s=s+abs(i)
			
		#fuj=0
		#n=0
		#for i in corela_list:
		#	j1=already_rated_items[n]
		#	fuj=fuj+i*(self.matrix[u,j1]-self.bais_matrix[u,j1])
		#	n=n+1
		print 's=%s' % s
		return self.bais_matrix[u,j]+ fuj/(2*s+1)



		

from operator import itemgetter
from numpy import nan

class Evaluation(object):
    """
    Base class for Evaluation

    It has the basic methods to load ground truth and test data.
    Any other Evaluation class derives from this base class.

    :param data: A list of tuples, containing the real and the predicted value. E.g: [(3, 2.3), (1, 0.9), (5, 4.9), (2, 0.9), (3, 1.5)]
    :type data: list
    """
    def __init__(self, data=None):
        #data is a list of tuples. E.g: [(3, 2.3), (1, 0.9), (5, 4.9), (2, 0.9), (3, 1.5)]
        if data:
            self._ground_truth, self._test = map(itemgetter(0), data), map(itemgetter(1), data)
        else:
            self._ground_truth = []
            self._test = []

    def __repr__(self):
        gt = str(self._ground_truth)
        test = str(self._test)
        return 'GT  : %s\nTest: %s' % (gt, test)
        #return str('\n'.join((str(self._ground_truth), str(self._test))))

    def load_test(self, test):
        """
        Loads a test dataset

        :param test: a list of predicted values. E.g: [2.3, 0.9, 4.9, 0.9, 1.5] 
        :type test: list
        """
        if isinstance(test, list):
            self._test = list(test)
        else:
            self._test = test

    def get_test(self):
        """
        :returns: the test dataset (a list)
        """
        return self._test

    def load_ground_truth(self, ground_truth):
        """
        Loads a ground truth dataset

        :param ground_truth: a list of real values (aka ground truth). E.g: [3.0, 1.0, 5.0, 2.0, 3.0]
        :type ground_truth: list
        """
        if isinstance(ground_truth, list):
            self._ground_truth = list(ground_truth)
        else:
            self._ground_truth = ground_truth

    def get_ground_truth(self):
        """
        :returns: the ground truth list
        """
        return self._ground_truth

    def load(self, ground_truth, test):
        """
        Loads both the ground truth and the test lists. The two lists must have the same length.

        :param ground_truth: a list of real values (aka ground truth). E.g: [3.0, 1.0, 5.0, 2.0, 3.0]
        :type ground_truth: list
        :param test: a list of predicted values. E.g: [2.3, 0.9, 4.9, 0.9, 1.5] 
        :type test: list
        """
        self.load_ground_truth(ground_truth)
        self.load_test(test)

    def add(self, rating, rating_pred):
        """
        Adds a tuple <real rating, pred. rating>

        :param rating: a real rating value (the ground truth)
        :param rating_pred: the predicted rating
        """
        if rating is not nan and rating_pred is not nan:
            self._ground_truth.append(rating)
            self._test.append(rating_pred)

    def add_test(self, rating_pred):
        """
        Adds a predicted rating to the current test list

        :param rating_pred: the predicted rating
        """
        if rating_pred is not nan:
            self._test.append(rating_pred)

    def compute(self):
        """
        Computes the evaluation using the loaded ground truth and test lists
        """
        if len(self._ground_truth) == 0:
            raise ValueError('Ground Truth dataset is empty!')
        if len(self._test) == 0:
            raise ValueError('Test dataset is empty!')

#Predictive-Based Metrics
class MAE(Evaluation):
    """
    Mean Absolute Error

    :param data: a tuple containing the Ground Truth data, and the Test data
    :type data: <list, list>
    """
    def __init__(self, data=None):
        super(MAE, self).__init__(data)

    def compute(self, r=None, r_pred=None):
        if r and r_pred:
            return round(abs(r - r_pred), ROUND_FLOAT)

        if not len(self._ground_truth) == len(self._test):
            raise ValueError('Ground truth and Test datasets have different sizes!')        

        #Compute for the whole test set
        super(MAE, self).compute()
        sum = 0.0 
        for i in range(0, len(self._ground_truth)):
            r = self._ground_truth[i]
            r_pred = self._test[i]
            sum += abs(r - r_pred)
        return round(abs(float(sum/len(self._test))), ROUND_FLOAT)

class RMSE(Evaluation):
    """
    Root Mean Square Error

    :param data: a tuple containing the Ground Truth data, and the Test data
    :type data: <list, list>
    """
    def __init__(self, data=None):
        super(RMSE, self).__init__(data)

    def compute(self, r=None, r_pred=None):
        if r and r_pred:
            return round(sqrt(abs((r - r_pred)*(r - r_pred))), ROUND_FLOAT)

        if not len(self._ground_truth) == len(self._test):
            raise ValueError('Ground truth and Test datasets have different sizes!')        

        #Compute for the whole test set
        super(RMSE, self).compute()
        sum = 0.0 
        for i in range(0, len(self._ground_truth)):
            r = self._ground_truth[i]
            r_pred = self._test[i]
            sum += abs((r - r_pred)*(r - r_pred))
        return round(sqrt(abs(float(sum/len(self._test)))), ROUND_FLOAT)
	
	


import sys
import codecs
import pickle
#from random import shuffle
from exceptions import ValueError
from numpy.random import shuffle


#from recsys.algorithm import VERBOSE

class Data:
    """
    Handles the relationshops among users and items
    """
    def __init__(self):
        #"""
        #:param data: a list of tuples
        #:type data: list
        #"""
        self._data = list([])

    def __repr__(self):
        s = '%d rows.' % len(self.get())
        if len(self.get()):
            s += '\nE.g: %s' % str(self.get()[0])
        return s

    def __len__(self):
        return len(self.get())

    def __getitem__(self, i):
        if i < len(self._data):
            return self._data[i]
        return None

    def __iter__(self):
        return iter(self.get())

    def set(self, data, extend=True):
        """
        Sets data to the dataset

        :param data: a list of tuples
        :type data: list
        """
        if extend:
            self._data.extend(data)
        else:
            self._data = data

    def get(self):
        """
        :returns: a list of tuples
        """
        return self._data

    def add_tuple(self, tuple):
        """
        :param tuple: a tuple containing <rating, user, item> information (e.g.  <value, row, col>)
        """
        #E.g: tuple = (25, "ocelma", "u2") -> "ocelma has played u2 25 times"
        if not len(tuple) == 3:
            raise ValueError('Tuple format not correct (should be: <value, row_id, col_id>)')
        value, row_id, col_id = tuple
        if not value and value != 0:
            raise ValueError('Value is empty %s' % (tuple,))
        if isinstance(value, basestring):
            raise ValueError('Value %s is a string (must be an int or float) %s' % (value, tuple,))
        if row_id is None or row_id == '':
            raise ValueError('Row id is empty %s' % (tuple,))
        if col_id is None or col_id == '':
            raise ValueError('Col id is empty %s' % (tuple,))
        self._data.append(tuple)

    '''def split_train_test(self, percent=80, shuffle_data=True):
        """
        Splits the data in two disjunct datasets: train and test

        :param percent: % of training set to be used (test set size = 100-percent)
        :type percent: int
        :param shuffle_data: shuffle dataset?
        :type shuffle_data: Boolean

        :returns: a tuple <Data, Data>
        """
        if shuffle_data:
            shuffle(self._data)
        length = len(self._data)
        train_list = self._data[:int(round(length*percent/100.0))]
        test_list = self._data[-int(round(length*(100-percent)/100.0)):]
        train = Data()
        train.set(train_list)
        test = Data()
        test.set(test_list)

        return train, test'''
    def split_K_fold(self, percent, shuffle_data=True):
        """
        Splits the data in two disjunct datasets: train and test

        :param percent: % of training set to be used (test set size = 100-percent)
        :type percent: int
        :param shuffle_data: shuffle dataset?
        :type shuffle_data: Boolean

        :returns: a tuple <Data, Data>
        """
	print 'alou'
	list=[]
        if shuffle_data:
            shuffle(self._data)
	length = len(self._data)
	for i in xrange(percent):
		
		train_list = self._data[:int(round(length*percent/100.0))]
		train = Data()
		train.set(train_list)
		list.append(train)
		self._data= self._data[int(round(length*percent/100.0)):]
       
       

        return list

    def load(self, path, force=True, sep='\t', format=None, pickle=False):
        """
        Loads data from a file

        :param path: filename
        :type path: string
        :param force: Cleans already added data
        :type force: Boolean
        :param sep: Separator among the fields of the file content
        :type sep: string
        :param format: Format of the file content. 
            Default format is 'value': 0 (first field), then 'row': 1, and 'col': 2.
            E.g: format={'row':0, 'col':1, 'value':2}. The row is in position 0, 
            then there is the column value, and finally the rating. 
            So, it resembles to a matrix in plain format
        :type format: dict()
        :param pickle: is input file in  pickle format?
        :type pickle: Boolean
        """
        
        if force:
            self._data = list([])
        if pickle:
            self._load_pickle(path)
        else:
            i = 0 
            for line in codecs.open(path, 'r', 'utf8'):
                data = line.strip('\r\n').split(sep)
                value = None
                if not data:
                    raise TypeError('Data is empty or None!')
                if not format:
                    # Default value is 1
                    try:
                        value, row_id, col_id = data
                    except:
                        value = 1
                        row_id, col_id = data
                else:
                    try:
                        # Default value is 1
                        try:
                            value = data[format['value']]
                        except KeyError, ValueError:
                            value = 1
                        try: 
                            row_id = data[format['row']]
                        except KeyError:
                            row_id = data[1]
                        try:
                            col_id = data[format['col']]
                        except KeyError:
                            col_id = data[2]
                        row_id = row_id.strip()
                        col_id = col_id.strip()
                        if format.has_key('ids') and (format['ids'] == int or format['ids'] == 'int'):
                            try:
                                row_id = int(row_id)
                            except:
                                print 'Error (ID is not int) while reading: %s' % data #Just ignore that line
                                continue
                            try:
                                col_id = int(col_id)
                            except:
                                print 'Error (ID is not int) while reading: %s' % data #Just ignore that line
                                continue
                    except IndexError:
                        #raise IndexError('while reading %s' % data)
                        print 'Error while reading: %s' % data #Just ignore that line
                        continue
                # Try to convert ids to int
                try:
                    row_id = int(row_id)
                except: pass
                try:
                    col_id = int(col_id)
                except: pass
                # Add tuple
                try:
                    self.add_tuple((float(value), row_id, col_id))
                   
                   
                except:
                    #if VERBOSE:
                        sys.stdout.write('\nError while reading (%s, %s, %s). Skipping this tuple\n' % (value, row_id, col_id))
                    #raise ValueError('%s is not a float, while reading %s' % (value, data))
                i += 1
               

    def _load_pickle(self, path):
        """
        Loads data from a pickle file

        :param path: output filename
        :type param: string
        """
        self._data = pickle.load(codecs.open(path))

    def save(self, path, pickle=False):
        """
        Saves data in output file

        :param path: output filename
        :type param: string
        :param pickle: save in pickle format?
        :type pickle: Boolean
        """
        if VERBOSE:
            sys.stdout.write('Saving data to %s\n' % path)
        if pickle:
            self._save_pickle(path)
        else:
            out = codecs.open(path, 'w', 'utf8')
            for value, row_id, col_id in self._data:
                try:
                    value = unicode(value, 'utf8')
                except:
                    if not isinstance(value, unicode):
                        value = str(value)
                try:
                    row_id = unicode(row_id, 'utf8')
                except:
                    if not isinstance(row_id, unicode):
                        row_id = str(row_id)
                try:
                    col_id = unicode(col_id, 'utf8')
                except:
                    if not isinstance(col_id, unicode):
                        col_id = str(col_id)

                s = '\t'.join([value, row_id, col_id])
                out.write(s + '\n')
            out.close()

    def _save_pickle(self, path):
        """
        Saves data in output file, using pickle format

        :param path: output filename
        :type param: string
        """
        pickle.dump(self._data, open(path, "w"))
