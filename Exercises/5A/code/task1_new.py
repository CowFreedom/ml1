from __future__ import division
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

n=20 #Anzahl der Stichproben
#alpha=2
#beta=1

numVariates=2 #number of basis functions

#Definitionen
test_range=np.linspace(-1,1,100)
sample_range=(npr.rand(n)*2)-1
noise_precision=(1/0.2)**2 #what happens if you change the input precision?
coefficients=[-0.3,0.5]
f = lambda x : coefficients[0]+coefficients[1]*x+0.2*np.random.normal(0,1,1)[0]

#data
data=np.array([f(x) for x in sample_range])

##helper function to build design Matrix
def phi_i(x,n):
    return x**n



#prior parameters
SIGMA_0=2*np.eye(numVariates)
MEAN_0=[0,0]



def plotprior(s):
	x, y = np.mgrid[-5:5:.01, -5:5:.01]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y
	#plt.subplot(s)
	radius=[]
	square=[]
	plt.axes().set_aspect('equal', 'box')
   
        #für Variablen w0, w1
	rv = multivariate_normal(MEAN_0, SIGMA_0)
        #Plotten der tatsächlichen Koeffizienten
        #w0,w1
	plt.xlabel('w0')        
	plt.ylabel('w1')
  
	area = [3.14159]
        
	plt.title('prior')
	
	plt.contourf(x, y, rv.pdf(pos))

    #plt.axis('box')
	plt.show()

#(x_s,t)=Teilmenge vom Inputtupel (x,t)
def plotposterior(x_s,t_s,s):
	Phi=np.array([[phi_i(x,k) for k in range(numVariates)] for x in x_s]) #Design Matrix
	SIGMA_n=npl.inv((npl.inv(SIGMA_0)+noise_precision*Phi.transpose().dot(Phi)))
	MEAN_n=noise_precision*SIGMA_n.dot(Phi.transpose().dot(t_s))
	#print(SIGMA_n)
	x, y = np.mgrid[-2:2:.01, -2:2:.01]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y
	#plt.subplot(s)
	radius=[]
	square=[]
	#plt.axes().set_aspect('equal', 'box')
   
        #für Variablen w0, w1
	rv = multivariate_normal(MEAN_n, SIGMA_n)
        #Plotten der tatsächlichen Koeffizienten
        #w0,w1
	#plt.xlabel('w0')        
	#plt.ylabel('w1')
  
	area = [3.14159]
        
	#plt.title('posterior')
	
	#plt.contourf(x, y, rv.pdf(pos))
	return(x,y,rv,pos)

def plotlikelihood(x_s,t_s,s):
	def L_x(w_0,w_1):
		sum=0

		for i in range(len(x_s)):
			sum=sum+(t_s[i]-w_1*x_s[i]-w_0)**2
		#print(sum)
		return np.exp(-0.5*sum)
	

	#plt.subplot(s)
	radius=[]
	square=[]
	#plt.axes().set_aspect('equal', 'box')

	# create data
 
	# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
	x=np.linspace(-2,2,100)
	y=np.linspace(2,-2,100)
	z=np.zeros(shape=(len(x),len(y)))
	iter=0

	for i in range(len(x)):
		for j in range(len(y)):
			z[j,i]=L_x(x[i],y[j])
										
	#print((t_s[0]-1.6*x_s[0]+0.3)**2+(t_s[1]-1.6*x_s[1]+0.3)**2)
	#print(L_x(-0.3,1.6))
	N = int(len(z)**.5)
#	z = z.reshape(N, N)
	
	#plt.imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	#cmap=cm.hot)
	#plotten des tatsächlichen means
	radius = [coefficients[0]]
	square = [coefficients[1]]	
	#plt.plot(radius, square, marker='+', linestyle='--', color='r', label='Square') 


	#plt.xlabel('w0')        
	#plt.ylabel('w1')
  
	area = [3.14159]
	
	#plt.title('posterior')

	return (x,y,z)
	#plt.contourf(x, y,L_x(pos))

    #plt.axis('box')
	#plt.show()
#plotprior(1)
#plotposterior(sample_range,data,2)
#plotlikelihood(sample_range,data,3)



def plotLikelihoodAll():
	radius = [coefficients[0]]
	square = [coefficients[1]]
	x=range(9)
	y=range(9)
	fig, ax = plt.subplots(3, 3,sharex=True, sharey=True)
	ax[0,0].set_title("Likelihood")
	#ax[0,0].axes().set_aspect('equal', 'box')
	(x,y,z)=plotlikelihood(sample_range[0:1],data[0:1],3)	
	ax[0,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[0,0].plot(radius, square, marker='+', linestyle='--', color='r', label='Square') 
	ax[0,0].set_ylabel('w1')
	ax[0,0].set_xlabel('w0')	
	(x,y,rv,pos)=plotposterior(sample_range[0:1],data[0:1],2)	
	ax[0,1].contourf(x, y, rv.pdf(pos))
	ax[0,1].set_title("Posterior")
	ax[0,2].plot(sample_range[0:1],data[0:1],linestyle=" ",marker="o", color="r")
	ax[0,2].set_title("Data")
	
	#second datapoint
	(x,y,z)=plotlikelihood(sample_range[0:2],data[0:2],3)	
	ax[1,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[1,0].plot(radius, square, marker='+', linestyle='--', color='r', label='Square') 
	(x,y,rv,pos)=plotposterior(sample_range[0:2],data[0:2],2)	
	ax[1,1].contourf(x, y, rv.pdf(pos))
	ax[1,2].plot(sample_range[0:2],data[0:2],linestyle=" ",marker="o", color="r")

	#20th datapoint
	(x,y,z)=plotlikelihood(sample_range[0:20],data[0:20],3)	
	ax[2,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[2,0].plot(radius, square, marker='+', linestyle='--', color='r', label='Square') 
	(x,y,rv,pos)=plotposterior(sample_range[0:20],data[0:20],2)	
	ax[2,1].contourf(x, y, rv.pdf(pos))
	ax[2,2].plot(sample_range[0:20],data[0:20],linestyle=" ",marker="o", color="r")
	
	plt.show()
	
def plotSingleLikelihoodAll():
	radius = [coefficients[0]]
	square = [coefficients[1]]
	x=range(9)
	y=range(9)
	fig, ax = plt.subplots(3, 3,sharex=True, sharey=True)
	ax[0,0].set_title("Single Likelihood")
	#ax[0,0].axes().set_aspect('equal', 'box')
	(x,y,z)=plotlikelihood(sample_range[0:1],data[0:1],3)	
	ax[0,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[0,0].plot(radius, square, marker='+', linestyle='--', color='r', label='Square') 
	ax[0,0].set_ylabel('w1')
	ax[0,0].set_xlabel('w0')	
	(x,y,rv,pos)=plotposterior(sample_range[0:1],data[0:1],2)	
	ax[0,1].contourf(x, y, rv.pdf(pos))
	ax[0,1].set_title("Posterior")
	ax[0,2].plot(sample_range[0:1],data[0:1],linestyle=" ",marker="o", color="r")
	ax[0,2].set_title("Data")
	
	#second datapoint
	(x,y,z)=plotlikelihood(sample_range[1:2],data[1:2],3)	
	ax[1,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[1,0].plot(radius, square, marker='+', linestyle='--', color='r', label='Square') 
	(x,y,rv,pos)=plotposterior(sample_range[0:2],data[0:2],2)	
	ax[1,1].contourf(x, y, rv.pdf(pos))
	ax[1,2].plot(sample_range[0:2],data[0:2],linestyle=" ",marker="o", color="r")

	#20th datapoint
	(x,y,z)=plotlikelihood(sample_range[18:19],data[18:19],3)	
	ax[2,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[2,0].plot(radius, square, marker='+', linestyle='--', color='r', label='Square') 
	(x,y,rv,pos)=plotposterior(sample_range[0:20],data[0:20],2)	
	ax[2,1].contourf(x, y, rv.pdf(pos))
	ax[2,2].plot(sample_range[0:20],data[0:20],linestyle=" ",marker="o", color="r")
	
	plt.show()
	
def plotQuiz():
	radius = [coefficients[0]]
	square = [coefficients[1]]
	x=range(9)
	y=range(9)
	fig, ax = plt.subplots(3, 3,sharex=True, sharey=True)
	ax[0,0].set_title("Single Likelihood")
	#ax[0,0].axes().set_aspect('equal', 'box')
	(x,y,z)=plotlikelihood(sample_range[0:1],data[0:1],3)	
	ax[0,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[0,0].set_ylabel('w1')
	ax[0,0].set_xlabel('w0')		
	(x,y,z)=plotlikelihood(sample_range[0:1],data[0:1],3)
	ax[0,1].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[0,1].set_title("Likelihood")
	ax[0,2].plot(sample_range[0:1],data[0:1],linestyle=" ",marker="o", color="r")
	ax[0,2].set_title("Data")
	
	#second datapoint
	(x,y,z)=plotlikelihood(sample_range[1:2],data[1:2],3)	
	ax[1,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot) 
	(x,y,z)=plotlikelihood(sample_range[0:2],data[0:2],3)
	ax[1,1].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[1,2].plot(sample_range[0:2],data[0:2],linestyle=" ",marker="o", color="r")

	#20th datapoint
	(x,y,z)=plotlikelihood(sample_range[18:19],data[18:19],3)	
	ax[2,0].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	(x,y,z)=plotlikelihood(sample_range[0:20],data[0:20],3)
	ax[2,1].imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
	cmap=cm.hot)
	ax[2,2].plot(sample_range[0:20],data[0:20],linestyle=" ",marker="o", color="r")
	
	plt.show()
	
plotLikelihoodAll()
#plotSingleLikelihoodAll()

#plotQuiz()
