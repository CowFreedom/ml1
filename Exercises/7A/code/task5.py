import matplotlib.pyplot as plt
from random import random
import numpy as np

class logistic_regression:
	def __init__(self,train):
		self.train=train
		self.w=np.random.rand(3) #three randomly drawn weights w_0, w11,w12
		print(self.w)
		self.calculateW()
		print(self.w)
		
	def calculateW(self):
		phi=lambda x: self.w[0]+self.w[1]*x[0]+self.w[2]*x[1]
		sigmoid=lambda x: 1/(1+np.exp(-phi(x)))
		def Err():
			sum=0
			for (x,t) in self.train:
				sum=sum+(sigmoid(x)-t)*np.array([1,x[0],x[1]])
			return sum*np.array([1,])
		iter=0
		alpha=1
		while iter<50:
			self.w=self.w-alpha*Err()
			iter=iter+1
	
	def predicted_label(self,x):
		phi=lambda x: self.w[0]+self.w[1]*x[0]+self.w[2]*x[1]
		sigmoid=lambda x: 1/(1+np.exp(-phi(x)))
		likelihood=sigmoid(x)
		if likelihood>=0.5:
			return 1
		else:
			return 0

def createData(dims, *positional_parameters, **keyword_parameters):
	if ('means' in keyword_parameters):
		return keyword_parameters['means']+np.random.randn(*dims)
	else:
		return np.random.randn(*dims)

def create2DDataNormal(n, mean1, mean2, cov1, cov2):
		return (np.random.multivariate_normal(mean1, cov1, int(0.5*n)),np.random.multivariate_normal(mean2, cov2, int(0.5*n)))		

def view_data_2D_example(n):
	X=createData((n,2))
	mean1=np.array([1,2])
	mean2=np.array([3,3])
	Sigma1=np.array([[0.5,0.3],[0.3,0.5]])
	Sigma2=np.array([[0.7,-0.3],[-0.3,0.5]])
	A, B=create2DDataNormal(100,mean1,mean2,Sigma1,Sigma2)
	plt.plot(A[:,0],A[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='Datenpunkte')
	plt.plot(B[:,0],B[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='Datenpunkte')
	plt.show()
	
#Shows a binary dataset with predicted labels and decision boundary after applying logistic regression	
def clustering_2D_example(n):
	mean1=np.array([1,2])
	mean2=np.array([3,3])
	Sigma1=np.array([[0.5,0.3],[0.3,0.5]])
	Sigma2=np.array([[0.7,-0.3],[-0.3,0.5]])
	A, B=create2DDataNormal(n,mean1,mean2,Sigma1,Sigma2)
	labelData=[(x,0) for x in A]
	labelData=labelData+[(x,1) for x in B]
	regressor=logistic_regression(labelData)
	
	#plot datapoints
	for x in A:
		predicted_label=regressor.predicted_label(x)
		plt.plot(x[0],x[1],marker='o', linestyle='', color=(predicted_label,0,0.5), label='Datenpunkte A')	

	for x in B:
		predicted_label=regressor.predicted_label(x)
		plt.plot(x[0],x[1],marker='^', linestyle='', color=(predicted_label,0,0.5), label='Datenpunkte B')	
			
	#plt.plot(A[:,0],A[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='Datenpunkte A')
	#plt.plot(B[:,0],B[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='Datenpunkte B')
	
	#plot separation line
	test_range=np.linspace(-2,5,2)
	plt.plot(test_range,[1/(regressor.w[2])*(-regressor.w[1]*x-regressor.w[0]) for x in test_range],marker=' ', linestyle='-', color=(random(),random(),random()), label='boundary')
	
	
	plt.show()

def nonlinear_boundary_2D_example(n):
	mean1=np.array([0,0])
	mean2=np.array([3,3])
	Sigma1=0.5*np.eye(2)
	Sigma2=np.array([[0.7,-0.3],[-0.3,0.5]])
	A, B=create2DDataNormal(100,mean1,mean2,Sigma1,Sigma2)
	plt.plot(A[:,0],A[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='Datenpunkte')
	plt.plot(B[:,0],B[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='Datenpunkte')
	test_range=np.linspace(-2,2,100)
	plt.plot(test_range,[np.sqrt(4-x**2) for x in test_range],marker=' ', linestyle='-', color=(0.5,0.5,0.5), label='boundary')
	plt.plot(test_range,[-np.sqrt(4-x**2) for x in test_range],marker=' ', linestyle='-', color=(0.5,0.5,0.5), label='boundary')	
	plt.show()
	
	
#view_data_2D_example(20)	
clustering_2D_example(190)
#nonlinear_boundary_2D_example(20)