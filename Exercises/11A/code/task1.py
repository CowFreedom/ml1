import matplotlib.pyplot as plt
from random import random
import numpy as np

class NeuralNet_2layer:
	def __init__(self,data):
		self.t=data[:,2]
		self.x=data[:,:2]
		self.w_node=np.random.rand(2,3) #two weights and one bias for the first layer
		self.w_node_out=np.random.rand(3) #two weights and one bias for the first layer
		self.backprop(self.x[0],self.t[0])
		print(self.eval([1,2]))
		#print(self.w_node_out)
		
	def eval(self,x):
		node1=self.tanh(self.w_node[0,0]*x[0]+self.w_node[0,1]*x[1]+self.w_node[0,2])
		node2=self.tanh(self.w_node[1,0]*x[0]+self.w_node[1,1]*x[1]+self.w_node[1,2])
		return self.logistic(self.w_node_out[0]*node1+self.w_node_out[1]*node2+self.w_node_out[2])
		
#	def optimize(self):
		#for i in range(len(self.t)):
			#self.backprop(self

	def backprop(self,x,t):
		for i in range(3):
			z=self.w_node_out[0]*x[0]+self.w_node_out[1]*x[1]+self.w_node_out[2]
			deriv=self.node_activation_derivative(z)
			loss=self.point_loss_derivative(self.logistic(z),t)
			self.w_node_out[i]+=loss*deriv*self.logistic(z)
		
		for i in range(2):
			for j in range(3):
				z=self.w_node[i,0]*x[0]+self.w_node[i,0]*x[1]+self.w_node[i,0]
				deriv=self.node_activation_derivative(z)
				
				
			
	def tanh(self, x):
		return np.tanh(x)

	def logistic(self, x):
		return 1/(1+np.exp(x))
	
	def point_loss(self,y,t):
		return 0.5*(y-t)**2
	
	def point_loss_derivative(self,y,t):
		return y-t
	
	def node_activation_derivative(self,x):
		return self.logistic(x)*(1-self.logistic(x))


		

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
	Sigma1=np.array([[0.9,0.3],[0.3,0.9]])
	Sigma2=np.array([[1.2,-0.4],[-0.4,1.2]])
	A, B=create2DDataNormal(n,mean1,mean2,Sigma1,Sigma2)
	A=np.hstack((A,0*np.ones(shape=(len(A),1))))
	B=np.hstack((B,np.ones(shape=(len(A),1))))
	data=np.vstack((A,B))
	model=NeuralNet_2layer(data)
	#print(A)
	#regressor=logistic_regression(labelData)
	plt.plot(A[:,0],A[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='class A')
	plt.plot(B[:,0],B[:,1],marker='o', linestyle='', color=(random(),random(),random()), label='class B')
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

	'''
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
	'''

	
#view_data_2D_example(20)	
clustering_2D_example(20)
#nonlinear_boundary_2D_example(20)