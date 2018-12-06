from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
import numpy as np
import math

n=300
m=8

d=8 #dimension of input data


sample_range=np.random.uniform(0,2,n) #range the data is spread around
sigma=np.random.rand(d,d)
x= np.random.multivariate_normal(np.array([0,1,2,3,4,5,6,7]), 10*np.eye(d)+0.5*(sigma+sigma.T), 5)

#dm=np.array([[sample_range[i]**j for j in range(1,m+1)] for i in range(n)], dtype='float')

#data=[sample_range.reshape(-1,1),[f(x)+np.random.normal(0,0.5) for x in sample_range]] #t values with added noise
			