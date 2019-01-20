# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt #zum plotten (remove)
#from scipy.stats import multivariate_normal

def plotdata(f,a,b,minValue,maxValue,num):
    test_range=np.linspace(minValue,maxValue,100)  
    
    #test_range=np.array([np.linspace(minValue, maxValue, 10), np.linspace(minValue, maxValue, 10)])
    
    fvalues=[f(x) for x in test_range]
    true_plot, = plt.plot(a, b,marker='o', linestyle='', color='r', label='Datenpunkte')
    #regression_plot, = plt.plot(test_range, fvalues, color='b', label='Regressionsfunktion')
    plt.title("Kovarianz="+str(np.cov(a,b,rowvar=False,ddof=0)[0][1]))
    plt.xlabel('x_'+str(num))        
    plt.ylabel('Weinklasse')
    #plt.axis('equal')
    #Kontrolliere Wertebereich der Axen
    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([0.5,3.5])
    
    plt.savefig('C:/Users/Tristan_local/OneDrive/myData/Machine Learning/Blatt 8/output'+str(num)+'.png', transparent=True)
  
    plt.show()




def heatmap(f):
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    radius=[]
    square=[]
    plt.axes().set_aspect('equal', 'box')

    plt.title('posterior')
    plt.plot(radius, square, marker='o', linestyle='--', color='r', label='Square')
    plt.contourf(x, y, g(pos))
    plt.savefig('C:/Users/Tristan_local/OneDrive/myData/Machine Learning/Blatt 8/output.eps', transparent=True)

    #plt.axis('box')
    plt.show    