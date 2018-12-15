from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
import numpy as np
import math

n=3000
#f=lambda x: 0.2+0.3*x+0.5*x**2+0.6*x**3+0.4*x**4+0.5*x**5+0.6*x**6+0.7*x**7
f=lambda x: 4+-2*x+2*x**2
k=1.0
#f=lambda x: 2+-k*3*x+(k**2)*4*x**2+-(k**3)*5*x**3+(k**4)*4*x**4+-(k**5)*4*x**5+(k**6)*6*x**6-(k**7)*5*x**7
#f=lambda x: 4+-2*x+2*x**2+-1*x**3+3*x**4
m=3

predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])
	
sample_range=np.random.uniform(0,25,n) #range the data is spread around
dm=np.array([[sample_range[i]**j for j in range(1,m+1)] for i in range(n)], dtype='float')

data=[sample_range.reshape(-1,1),[f(x)+np.random.normal(0,0.5) for x in sample_range]] #t values with added noise
			
clf = linear_model.Lasso(alpha=0.1, max_iter=1000)
#clf = linear_model.LinearRegression()
clf.fit(dm, data[1])
#lr = LinearRegression()
#lr.fit(data[0], data[1])

test_range=np.linspace(0,2,100)

def y(x):
	sum=0
	for i in range(m):
		sum=sum+clf.coef_[i]*(x**(i+1))
	return sum+clf.intercept_
print("Weights",clf.coef_)
#plt.plot(test_range,[f(x) for x in test_range], label="True plot")
#plt.plot(sample_range,data[1],linestyle=" ", marker="o")
#plt.plot(test_range,[y(x) for x in test_range], label="True plot")
#plt.show()

print("intercept:" ,clf.intercept_)


#Calculation of design matrix
dm2=np.array([[sample_range[i]**j for j in range(m+1)] for i in range(n)], dtype='float')



eps = 5e-3 
alphas_lasso, _, coefs = linear_model.lars_path(dm2, np.array(data[1]), method='lasso', verbose=True)
#print(coefs)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]
neg_log_alphas_lasso = -np.log10(alphas_lasso)
for i in range(m):
	plt.plot(xx, coefs[i],label="w"+str(i))
	
legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')	
#plt.plot(xx, coefs.T)
#plt.plot(np.array([1,2,3]),np.array([[4,9],[5,9],[6,9]]))
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / |max_likelihood(coef)|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')


plt.show()
