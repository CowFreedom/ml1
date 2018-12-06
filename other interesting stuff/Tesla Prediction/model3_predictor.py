import numpy as np
import matplotlib.pyplot as plt #zum plotten (remove)
import math
import numpy.linalg as npl
import matplotlib.patches as mpatches

prodNumbers=np.array([[0,1,2,3,4],
[260,2425,9766,28578,53239]])


def plotProdNumbers(x):
	plt.plot(prodNumbers[0],prodNumbers[1], linestyle=" ", marker="o", color=(0.8,0,0), label='Square')
	plt.title("Model 3 Production numbers")
	plt.xlabel('Quarter')        
	plt.ylabel('Numbers')
	
def runLassoRegressionExp(data,test_range):
	##Generiere noisebehaftete Datensätze
	mu=0.0
	sigma=0.3
	basis_functions=30 #Anzahl der Basisfunktionren
	regularizer=0.1#regularizer

	##Generierung unserer Polynomkoeffizienten
	def phi_i(x,k):
		if (k==0):
			return 1
		else:
			mu_i=np.linspace(min(test_range),max(test_range),(basis_functions-1))[k-1]
			s=1.3 ##änderbar
			return math.exp((-math.pow((x-mu_i),2))/(2*math.pow(s,2)))

	#Matrix groß PhWWi
	Phi=np.array([[phi_i(x,k) for k in range(basis_functions)] for x in data[0]]) 

	#Matrix groß Sigma
	w=np.array(((npl.pinv(regularizer*np.identity(basis_functions)+np.transpose(Phi).dot(Phi))).dot(np.transpose(Phi).dot(data[1]))))

	###Generierung unserer Regressionsfunktion
	def y_x(x):
		phi_temp=np.array([phi_i(x,k) for k in range(basis_functions)])
		return w.dot(phi_temp)

	regression_values=[y_x(k) for k in test_range]
	plt.plot(test_range,regression_values,color=(0.2,0.3,0.6), label="Line 2")
	
	red_patch = mpatches.Patch((0.2,0.3,0.6), label='Exponential Basis functions with lasso')

	plt.legend(handles=[red_patch])	

def runLassoRegressionPoly(data,test_range):
	##Generiere noisebehaftete Datensätze
	mu=0.0
	sigma=0.3
	basis_functions=10 #Anzahl der Basisfunktionren
	regularizer=1#regularizer

	##Generierung unserer Polynomkoeffizienten
	def phi_i(x,k):
		return x**k

	#Matrix groß PhWWi
	Phi=np.array([[phi_i(x,k) for k in range(basis_functions)] for x in data[0]]) 

	#Matrix groß Sigma
	w=np.array(((npl.pinv(regularizer*np.identity(basis_functions)+np.transpose(Phi).dot(Phi))).dot(np.transpose(Phi).dot(data[1]))))

	###Generierung unserer Regressionsfunktion
	def y_x(x):
		phi_temp=np.array([phi_i(x,k) for k in range(basis_functions)])
		return w.dot(phi_temp)

	regression_values=[y_x(k) for k in test_range]
	plt.plot(test_range,regression_values,color=(0.2,0.6,0.3),label="Line 1")
	
	blue_patch = mpatches.Patch((0.2,0.6,0.3), label='Polynomial Basis functions with lasso')

	#plt.legend(handles=[blue_patch])

def runPredictiveDist(data, test_range):
	alpha=3 #precision
	beta=0.3



	m=11
	##Generierung unserer Polynomkoeffizienten
	def phi_i(x,k):
		if (k==0):
			return 1
		else:
			mu_i=np.linspace(min(test_range),max(test_range),(m-1))
			s=1.3 ##änderbar
			return math.exp(-math.pow((x-mu_i)[k-1],2)/(2*math.pow(s,2)))
			
	def Phi_n(x):
		return np.array([phi_i(x,k) for k in range(m)])

	#Matrix groß Phi
	Phi=np.array([[phi_i(x,k) for k in range(m)] for x in data[0]]) 


	#Matrix groß Sigma

	SIGMA_n=npl.pinv((1/(alpha))*np.identity(m)+beta*np.transpose(Phi).dot(Phi))
	MEAN_n=beta*(SIGMA_n.dot(np.transpose(Phi).dot(data[1])))

	def VAR_n(x):
		return (1/beta)+np.transpose(Phi_n(x)).dot(SIGMA_n).dot(Phi_n(x))
	y1values=[np.transpose(MEAN_n).dot(Phi_n(x))-math.pow(VAR_n(x),5.5) for x in test_range]
	y2values=[np.transpose(MEAN_n).dot(Phi_n(x))+math.pow(VAR_n(x),5.5) for x in test_range]
	plt.plot(test_range,y1values,test_range,y2values,color='black')
	plt.fill_between(test_range,y1values,y2values,color='red',alpha='0.5')
	blue_patch = mpatches.Patch((0,0,0), label='Exponential Kernel with predictive distribution')
	plt.legend(handles=[blue_patch])
		
plotProdNumbers(prodNumbers)
#runLassoRegressionExp(prodNumbers,np.linspace(0,8,20))
#runLassoRegressionPoly(prodNumbers,np.linspace(0,10,30))
runPredictiveDist(prodNumbers,np.linspace(0,5,200))
plt.show()