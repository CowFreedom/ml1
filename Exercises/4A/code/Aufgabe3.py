# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import math

#Einstellbar
n=15 #Anzahl der Stichproben
basis_functions=25
maxXvalue=1#Es werden Stichproben gezogen in der Region [0,maxXvalue]
minXvalue=0


#Definitionen
testpoints=100
test_range=np.linspace(minXvalue,maxXvalue,testpoints)
sample_range=np.random.uniform(minXvalue,maxXvalue,n)

#generiere Funktion f(x,a)
def f_x(x):
 return math.sin(2*math.pi*x)
 
def f_x_noise(x):
 return math.sin(2*math.pi*x)+np.random.normal(mu,sigma,1)[0]
  
##Generiere noisebehaftete Datensätze
mu=0.0
sigma=0.3
iterations=10
value_container=[]
bias_container=[]

m=1
regularizer_container=np.zeros(m)
bias=np.zeros(m)
variance=np.zeros(m)
##############
#Berechnung der Maßzahlen
##############

def calc_avg_prediction(y_container,index):
	sum=0
	for u in range(len(y_container)):
		sum=sum+y_container[u][index]
	return (1/len(y_container))*sum

def calc_bias_squared(f,y_container):
	
	n=len(sample_range)
	sum=0
	for index in range(len(sample_range)):
		sum=sum+math.pow(calc_avg_prediction(y_container,index)-f_x(sample_range[index]),2)
	return (1/n)*sum

def calc_variance_squared(f,y_container):
	n=len(y_container[0])
	L=len(y_container)
	#print("N",y_container[0,0])
	sum=0
	for i in range(n):
		for j in range(L):
			sum=sum+math.pow(y_container[j][i]-calc_avg_prediction(y_container,i),2)
	return (1/n)*(1/L)*sum
regularizer_container[0]=0.1
regularizer_container=np.linspace(-1,100,m)	
#Berechnung der Regression
for j in range(m):

	#regression_values_container=np.zeros(shape=(iterations,len(sample_range)))
	#data_values_container=np.zeros(shape=(iterations,len(sample_range)))
	for i in range(iterations):
		data=[f_x_noise(x) for x in sample_range]  
		##Generierung unserer Polynomkoeffizienten
		def phi_i(x,k):
			if (k==0):
				return 1
			else:
				mu_i=np.linspace(minXvalue,maxXvalue,(basis_functions-1))[k-1]
				s=0.3 ##änderbar
				return math.exp((-math.pow((x-mu_i),2))/(2*math.pow(s,2)))

		#Matrix groß Phi
		Phi=np.array([[phi_i(x,k) for k in range(basis_functions)] for x in sample_range]) 

		#Matrix groß Sigma
		w=np.array(((npl.pinv(1*regularizer_container[j]*np.identity(basis_functions)+np.transpose(Phi).dot(Phi))).dot(np.transpose(Phi).dot(data))))

		###Generierung unserer Regressionsfunktion
		def y_x(x):
			phi_temp=np.array([phi_i(x,k) for k in range(basis_functions)])
			return w.dot(phi_temp)

		#regression_values_container[i]=np.array([y_x(k) for k in sample_range])
		value_container.insert(i,[y_x(k) for k in test_range])#werte für die testrange
		#data_values_container[j]=data
	bias[j]=calc_bias_squared(f_x,value_container)
	variance[j]=calc_variance_squared(f_x,value_container)





print(variance)
#print bias variance decomposition
#line_up ,= plt.plot(regularizer_container,variance,label="variance",linestyle="-",marker="o")
#line_down, =plt.plot(regularizer_container,bias,label="Bias^2",linestyle="-",marker="o")
#plt.legend(handles=[line_up, line_down])
#plt.xlabel(r'$\lambda=$')        
#plt.ylabel(' ')			

#print functions
fvalues=[f_x(k) for k in test_range]
f_noise_values=[f_x_noise(k) for k in sample_range]
for i in range(iterations):
    plt.plot(test_range,value_container[i], alpha=0.7)
plt.plot(sample_range,f_noise_values,linestyle=" ", marker="o")    
plt.plot(test_range,fvalues, label='Test', alpha=0.7)
#plt.title(r'$\lambda=$'+str(regularizer_container[0])+',n='+str(n)+',basisfunctions='+str(basis_functions))
#plt.xlabel('x')        
#plt.ylabel('t')
plt.show()