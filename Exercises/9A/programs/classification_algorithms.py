
import numpy as np
from random import random
from collections import Counter
import sys
#Let us build a linear classifier. Because bivariate logistic regression is only
#a subcase of multivariate regression, are only concerned with the latter.
class multi_logistic_regression:
	def __init__(self,X,t,dim):
		self.X=X
		self.t=t
		self.dimT=dim
		#self.w=np.random.rand(self.dimT,len(X[0])) #weights are randomly drawn at first
		self.w=1*np.ones(shape=(self.dimT,len(X[0])))
		self.pivot=self.dimT-1
		self.counts=Counter(t)
		self.calculateW()
		
	def calculateW(self):
		softmax_pivot=lambda x: 1/(1+np.sum(np.exp(-self.w[0:self.pivot].dot(x))))
		inv_softmax_pivot=lambda x: 1+np.sum(np.exp(-self.w[0:self.pivot].dot(x)))
		#softmax_pivot=lambda x: np.exp(-self.w[self.pivot-1].dot(self.X[0]))/np.sum(self.w[0:self.pivot].dot(self.X[0]))
		#print("w: ",self.w)
		def derivative_ln_Err():
			p=self.w[self.pivot] 
			sum1=0
			for i in range(len(self.X)):
				if (self.t[i]==(self.pivot)):
					sum1=sum1+self.X[i]
					#print("I'm in")
			result=np.zeros(shape=(self.dimT,len(self.X[0])))
			for i in range(self.pivot):
				sum2=0
				for j in range(len(self.X)):
					if (self.t[j]==i):	
						#sum2=sum2-self.X[j]*np.exp(-self.w[i].dot(self.X[j]))*(softmax_pivot(self.X[j]))#diff_wi
						sum2=sum2-self.X[j]*np.exp(-self.w[i].dot(self.X[j]))*softmax_pivot(self.X[j])
					if self.t[j]==(self.pivot):
						#sum2=sum2+self.X[j]*np.exp(-self.w[i].dot(self.X[j]))*(softmax_pivot(self.X[j]))
						sum2=sum2+self.X[j]*np.exp(-self.w[i].dot(self.X[j]))*softmax_pivot(self.X[j])
				result[i]= sum2
			result[self.pivot]=sum1
			
			return result
		#print("Err:",derivative_ln_Err())			
		#sys.exit()
		#print(2*derivative_ln_Err(self.X[0]))
		#return 

		iter=0
		alpha=1
		while iter<1000:
			w_h=derivative_ln_Err()
			for i in range(self.dimT):
				self.w[i]=self.w[i]+alpha*w_h[i]
			
			iter=iter+1
			#print("isZero?",derivative_ln_Err())
			print(self.w)

		
	
	def predicted_label(self,X):
		softmax_pivot=lambda x: 1/(1+np.sum(np.exp(-self.w[0:self.pivot].dot(x))))
		likelihood_classes=np.zeros(self.dimT)
		#print(self.w.shape)
		#print(X.shape)
		for i in range(self.dimT-1):
			likelihood_classes[i]=np.exp(-self.w[i].dot(X))*softmax_pivot(X)
			#print(-self.w[self.pivot].dot(X))
		likelihood_classes[self.pivot]=softmax_pivot(X)
		#print("LIKELIHOOD",likelihood_classes)
		#print("Sum Likelihood",np.sum(likelihood_classes))
		return (np.argmax(likelihood_classes),max(likelihood_classes))


#The design matrix in the training phase should be the same
#as in the testing phase.
#If you change this function you have to equally change
#the design matrix function in the "sample.submission.py".
	
def buildLinearDesignMatrix(data):
	X=np.ones(shape=(len(data),len(data[0])+1))
	
	for i in range(len(data)):
		X[i,1:,]=data[i]
	return X
	
#X=buildPolynomialDesignMatrix(trainX,1)

