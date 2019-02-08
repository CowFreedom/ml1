import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


n=20
A1= np.random.multivariate_normal([-1,-1], [[0.05,-0.02],[-0.02,0.05]], 20)
A2= np.random.multivariate_normal([1,1], [[0.05,-0.02],[-0.02,0.05]], 20)
B=np.random.multivariate_normal([0,0], 0.02*np.eye(2), 20)
plt.plot(A1[:,0],A1[:,1],linestyle=" ",marker="x")
plt.plot(A2[:,0],A2[:,1],linestyle=" ",marker="x")
plt.plot(B[:,0],B[:,1],linestyle=" ",marker="o")
plt.xlabel("x1")
plt.xlabel("x2")
plt.show()