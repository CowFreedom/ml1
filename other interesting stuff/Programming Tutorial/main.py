import numpy as np
import math
import matplotlib.pyplot as plt
import diagnostics as ds
import linear_regression as lr
import copy
def generateExponentialDesignMatrix(data,minX,maxX,m):
	s=100
	f=lambda x, mu: math.exp(-0.5*s*(x-mu)**2)
	n=len(data)


	dm=np.ones(shape=(n,m), dtype='float')
	mus=np.linspace(minX,maxX,m-1)
	for j in range(n):
		for i in range(1,m-1):
			dm[j,i]=f(data[j],mus[i])
	def dv(x): 
		dv=np.ones(shape=(m))
		for i in range(1,m):
			dv[i]=math.exp(-0.5*s*(x-mus[i-1])**2)
		return dv
	return (dm,dv)
	
def generateSinosoidalDesignMatrix(data,minX,maxX,m):
	s=1*math.pi
	f=lambda x, mu: math.sin(s*(x-mu)**2)
	n=len(data)


	dm=np.ones(shape=(n,m), dtype='float')
	mus=np.linspace(minX,maxX,m-1)
	for j in range(n):
		for i in range(1,m-1):
			dm[j,i]=f(data[j],mus[i])
	def dv(x): 
		dv=np.ones(shape=(m))
		for i in range(1,m):
			dv[i]=math.sin(s*(x-mus[i-1])**2)
		return dv
	return (dm,dv)

def plotLinearRegressionExample():

	#Generating regression data

	n=30 #number of data points

	f=lambda x: math.sin(2*math.pi*x) #true functon t=f(x)
	sample_range=np.random.uniform(0,2,n) #range the data is spread around
	data=np.array([sample_range,[f(x)+np.random.normal(0,1) for x in sample_range]]) #t values with added noise
	#Generating model#
	m=6 #number of basis functions
	reg=0.1 #value of regularizer lambda
	model=lr.linear_regression(data) #generate linear regression model
	model.train(m,lasso='l2',reg=reg) #train the model (calculate weights w)

	#Plot predictions
	test_range=np.linspace(0,2,100) #for plotting the predictions	
	predictions=model.predict(test_range) #t values of predictions

	plt.plot(data[0,:],data[1,:],linestyle=" ", marker="o")
	plt.plot(test_range,predictions)
	plt.show()
	
def plotLinearRegressionExponentialExample():

	#Generating regression data

	n=30 #number of data points

	f=lambda x: math.sin(2*math.pi*x) #true functon t=f(x)
	sample_range=np.random.uniform(0,2,n) #range the data is spread around
	data=np.array([sample_range,[f(x)+np.random.normal(0,0.6) for x in sample_range]]) #t values with added noise
	#Generating model#
	m=25 #number of basis functions
	reg=0.1 #value of regularizer lambda
	(dm,dv)=generateExponentialDesignMatrix(sample_range,0,2,m)
	model=lr.linear_regression(data) #generate linear regression model
	model.train(m,lasso='l2',reg=reg, design_matrix=dm, basis_vector=dv) #train the model (calculate weights w)

	#Plot predictions
	test_range=np.linspace(0,2,100) #for plotting the predictions	
	predictions=model.predict(test_range) #t values of predictions

	plt.plot(data[0,:],data[1,:],linestyle=" ", marker="o")
	plt.plot(test_range,predictions)
	plt.show()

def plotLinearRegressionSineExample():

	#Generating regression data

	n=30 #number of data points

	f=lambda x: math.sin(2*math.pi*x) #true functon t=f(x)
	sample_range=np.random.uniform(0,2,n) #range the data is spread around
	data=np.array([sample_range,[f(x)+np.random.normal(0,0.6) for x in sample_range]]) #t values with added noise
	#Generating model#
	m=25 #number of basis functions
	reg=0.1 #value of regularizer lambda
	(dm,dv)=generateSinosoidalDesignMatrix(sample_range,0,2,m)
	model=lr.linear_regression(data) #generate linear regression model
	model.train(m,lasso='l2',reg=reg, design_matrix=dm, basis_vector=dv) #train the model (calculate weights w)

	#Plot predictions
	test_range=np.linspace(0,2,100) #for plotting the predictions	
	predictions=model.predict(test_range) #t values of predictions

	plt.plot(data[0,:],data[1,:],linestyle=" ", marker="o")
	plt.plot(test_range,predictions)
	plt.show()	
	
def plotBiasCovarianceDecomposition():


	#Generating regression data

	n=2000 #number of data points
	k=10 #number of regression functions
	m=25 #number of basis functions
	reg=0 #value of regularizer lambda
	f=lambda x: math.sin(2*math.pi*x) #true functon t=f(x)
	
	regularizer_range=np.linspace(0,3,1000)
	bias_values=[]
	variance_values=[]
	for u in regularizer_range:
		test_range=np.linspace(0,2,100) #range where we test the regression unctions
		functions=[]
		for i in range(k):
			sample_range=np.random.uniform(0,2,n) #range the data is spread around
			data=np.array([sample_range,[f(x)+np.random.normal(0,1) for x in sample_range]]) #t values with added noise
			
			#Generating model#
			model=lr.linear_regression(data) #generate linear regression model
			model.train(m,lasso='l2',reg=u) #train the model (calculate weights w)
			functions.append(model)
		measures=ds.bias_variance_linear_regression(functions,f)
		bias_values.append(measures.bias_squared_regression(test_range))
		variance_values.append(measures.variance_regression(test_range))
		#print(measures.bias_squared_regression(test_range))
	plt.plot(regularizer_range,bias_values,linestyle="-", marker=" ", label= "Bias^2")
	plt.plot(regularizer_range,variance_values,linestyle="-", marker=" ", label ="Variance")
	plt.xlabel('lambda')
	plt.ylabel('Error')
	plt.legend(loc='best', frameon=False)
	plt.show()	

def plotBiasCovarianceExponentialDecomposition():


	#Generating regression data

	n=30 #number of data points
	k=10 #number of regression functions
	m=25 #number of basis functions
	reg=0 #value of regularizer lambda
	f=lambda x: math.sin(2*math.pi*x) #true functon t=f(x)
	functions=[]
	regularizer_range=np.linspace(-5,5,10)
	bias_values=[]
	for u in regularizer_range:
		test_range=np.linspace(0,2,100) #range where we test the regression unctions
		for i in range(k):
			sample_range=np.random.uniform(0,2,n) #range the data is spread around
			data=np.array([sample_range,[f(x)+np.random.normal(0,2) for x in sample_range]]) #t values with added noise
			
			#Generating model#
			model=lr.linear_regression(data) #generate linear regression model
			(dm,dv)=generateExponentialDesignMatrix(sample_range,0,2,m)
			model.train(m,lasso='l2',reg=u, design_matrix=dm, basis_vector=dv) #train the model (calculate weights w)
			functions.append(model)
		measures=ds.bias_variance_linear_regression(functions,f)
		bias_values.append(measures.bias_squared_regression(test_range))
		#print(measures.bias_squared_regression(test_range))
	plt.plot(regularizer_range,bias_values,linestyle="-", )
	plt.show()	
	#Plot predictions
	

#plotLinearRegressionExample()
#plotLinearRegressionExponentialExample()
#plotLinearRegressionSineExample()
plotBiasCovarianceDecomposition()
#plotBiasCovarianceExponentialDecomposition()