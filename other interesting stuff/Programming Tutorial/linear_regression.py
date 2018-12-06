import numpy as np
import sys
import math
import matplotlib.pyplot as plt
class linear_regression:
	def __init__(self,data):
		self.x=data[0,:]
		self.y=data[1,:]
		self.n=len(data[0]) #length of the data
		
	'''
	m= number of basis functions
	
	If the optional parameter "design_matrix" is supplied, 
	an external design matrix will be used.
	
	Otherwise, a design matrix with m polynomial basis functions will be
	used.
	'''
	def train(self, m, *positional_parameters, **keyword_parameters):
		if ('design_matrix' in keyword_parameters and 'basis_vector' in keyword_parameters):
			self.dm=keyword_parameters['design_matrix']
			self.dv=keyword_parameters['basis_vector']
	
			if (self.dm.shape != (self.n,m)): #if user supplied design matrix has wrong dimension crash program
				raise ValueError('User supplied design matrix has wrong dimensions.')
		else:
			self.dm=np.array([[self.x[i]**j for j in range(m)] for i in range(self.n)], dtype='float')
			self.dv=lambda x: np.array([math.pow(x,i) for i in range(m)])
		self.w=np.zeros(shape=(m,1))
		
		if ('lasso' in keyword_parameters and 'reg' in keyword_parameters):
			if keyword_parameters['lasso']=='l2':
				self.__l2_regression(keyword_parameters['reg'])
			
	#m=number of basis function, lambda=regularization parameter
	def __l2_regression(self,reg):
		n,m=self.dm.shape
		#print("reg",reg)
		self.w=np.linalg.solve(reg*np.eye(m)+self.dm.T.dot(self.dm),self.dm.T.dot(self.y))

	def predict(self, x):
		return np.array([self.w.dot(self.dv(xi)) for xi in x])

		
class bayesian_linear_regression:	
	def __init__(self,data):
		self.x=data[0,:]
		self.y=data[1,:]
		self.n=len(data[0]) #length of the data

'''	
n=3
sample_range=np.random.uniform(0,2,n)

f=lambda x: math.sin(2*math.pi*x)+np.random.normal(0,1)
data=np.array([sample_range,[f(x) for x in sample_range]])
phi=np.eye(3)

model=linear_regression(data)
model.train(2,lasso='l2',reg=0)

test_range=np.linspace(0,2,100)	
predictions=model.predict(test_range)

plt.plot(data[0,:],data[1,:],linestyle=" ", marker="o")
plt.plot(test_range,predictions)
plt.show()

#print(model.dm)
'''