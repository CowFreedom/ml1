import numpy as np
from scipy.stats import gamma
from scipy.stats import t
import matplotlib.pyplot as plt

def log_studt_pdf(d):
	v=1
	sigma=1
	return -0.5*(v+1)*np.log((1+(1/v)*((d**2)/sigma**2)))+np.log((gamma.pdf(0.5*(v+1),1))-np.log((np.sqrt(v*np.pi*sigma**2*gamma.pdf(0.5*v,1)))))

def log_norm_pdf(d):
	sigma=1
	return -np.log(np.sqrt(2*np.pi*sigma**2))-0.5*(d**2)/(sigma**2)

testrange=np.linspace(-5,5,100)

plt.plot(testrange,[log_studt_pdf(x) for x in testrange], color="red", label="student t pdf")
plt.plot(testrange,[log_norm_pdf(x) for x in testrange], color="grey",label='gaussian pdf')
#plt.plot(testrange,np.log(t.pdf(testrange,1)),label='student t pdf')
plt.xlabel('t-w^(t)x')
plt.ylabel('log density)')
plt.legend(loc='best', frameon=False)
plt.show()