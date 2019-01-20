# -*- coding: utf-8 -*-
import numpy as np
import data as dc
import imp
import numpy.linalg as npl
import plotter as pt
import math#delete later
import numpy.random as npr#delete later
import sys
import random

imp.reload(dc)

imp.reload(pt)

class linear_regression_algorithms:
    

    
    #Simple lineare Regression mit eindimensionalem Definitions- sowie Wertebereich
    @staticmethod
    def simple_linear_regression(x1dim, y1dim):
        beta1=linear_regression_algorithms.pearson_product_moment(x1dim,y1dim)
        beta0=np.mean(y1dim)-(beta1*np.mean(x1dim))
        
        #Lineare Funktion, die zurückgegeben wird
        def linear_function(x):
            return beta1*x+beta0 
                
        return linear_function
        
      
    #Berechnet das Pearson Produkt Moment. Wichtig: Bei der Kovarianzberechnung unbedingt ddof auf null
    #setzen, da wir eine Population vor uns haben und kein Sample (default ist ddof=1) 
    @staticmethod  
    def pearson_product_moment( x1dim, y1dim):
        return np.cov(x1dim,y1dim,rowvar=False,ddof=0)[0][1]/(np.std(y1dim)*np.std(x1dim))

    #Curve Fitting über polynomielle Basisfunktionen
    #Settings[0]=[Anzahl der Basisfunktionen]
    @staticmethod 
    def polynomial_basis_functions(dataobject, settings):
        m=settings[0][0]#Anzahl der Basisfunktionen
    
        def phi_k(x,k):
            if (k==0):
                return 1
            else:
                return x**k
                
        Phi=np.array(linear_regression_algorithms.Phi_n(dataobject,m,lambda x: x))
        
        w=npl.pinv(np.transpose(Phi).dot(Phi)).dot(np.transpose(Phi).dot(dataobject.getYdata()))
        #w=npl.pinv(np.transpose(Phi).dot(Phi))
        #w=np.transpose(Phi)
        #w=[] 
        def regression_polynom(x):
            tempdata=dc.data_class()
            #Teste, ob der Input auch in Listenform ankommt. Wenn nicht, modifiziere ihn
            if(isinstance(x,(list,np.ndarray))==True):
                tempdata.changeXdata([x])
            elif isinstance(x,( int, float ))==True:
                tempdata.changeXdata([[x]])   
                
            temp=0
            tempPhi=linear_regression_algorithms.Phi_n(tempdata,m,lambda x: x )
            if (len(w)!=len(tempPhi[0])):
                print("Dimensionen der Eingabedaten stimmten nicht mit den Polynomkoeffizienten überein."+str(tempdata.x_data))
                sys.exit(0)
            else:
                for i in range (len(w)):
                    temp=temp+w[i]*tempPhi[0][i]
                                   
                return temp
        tempdata=dc.data_class()
        tempdata.changeXdata([[2,3]])
        temp=0
        tempPhi=linear_regression_algorithms.Phi_n(tempdata,m,lambda x: x )       
        
        return regression_polynom
      
     #Berechnet die Matrix Phi
    @staticmethod   
    def Phi_n(dataobject, m, phi_k, *args, **optional_functions):
        if (m==0):
            return []
        Phi=[]

        #Erstelle dynamischen for-loop
        
        for i in range(len(dataobject.getXdata())):
            temp=[]
            for k in range(m-1): #bis m-1, weil die Konstante Basisfunktion schon unten per default hinzugefügt wird
                if ('weight_function' in optional_functions):
                    temp+=linear_regression_algorithms.dynamic_sum(dataobject,k+1, i, optional= optional_functions['weight_function'])
                else:
                    temp+=linear_regression_algorithms.dynamic_sum(dataobject,k+1, i, optional= (lambda x: x))                    
               # print("k:"+str(k))
            
            temp.insert(0,1) #Weil phi_k(x,0)=1 bei uns
            Phi.append(temp)
        
        return  Phi
    
    
    @staticmethod
    def dynamic_sum(dataobject,m, index,*args, **optional_functions):

        Phi=[]
        d=dataobject.getXdimension()        
        r=[0]*m #Jeder der m-Einträge symbolisiert einen loop
        schranke=False
        Phi_n_row=[]
        while (schranke==False):
            total=0
            
            for k in range(m):
                    total+=r[k]
            
            if (total>=(d-1)*m ):
                   # print("BING")
                    schranke=True
            
            #if iter==d**d:
                #schranke=True
                
            change=True
        
            k=m-1 #innerster Loop in der Kaskade
          
               
            while (change==True and k>=0):
                r[k]=r[k]+1
                if (r[k]>d-1):
                    r[k]=0
                    change=True
                                  

                else:
                    change=False
                

                k=k-1
            
            temp=1
            for i in range(m):
                if ('optional' in optional_functions):
                    temp=temp*optional_functions['optional'](dataobject.getXdata()[index][r[i]])
                else:
                    temp=temp*dataobject.getXdata()[index][r[i]]
            
            if ('optional2' in optional_functions):
                Phi_n_row.append(optional_functions['optional2'](temp)) 
            else:
                Phi_n_row.append((temp)) 
            #print(str(r[0])+" temp:"+str(temp))
                                                            
        return Phi_n_row
                                
                                                                            
    @staticmethod    
    def regressiontype(number, dataobject, settings):
        regressiontype={0:linear_regression_algorithms.polynomial_basis_functions, 1:linear_regression_algorithms.simple_linear_regression}
        if (settings[10][0]==1):
            datacollection=linear_regression_algorithms.k_fold_validation(dataobject)
            f=regressiontype[number](datacollection[0],settings)
            print("R^2 after 2-fold testing:",linear_regression_algorithms.r_squared(datacollection[1], f))      
            return f
        else:
            return regressiontype[number](dataobject,settings)
        

    @staticmethod
    def r_squared(dataobject, regression_function):
        meanY=np.mean(dataobject.getYdata())
        temp1=0
        temp2=0
        for i in range(len(dataobject.getYdata())):
            temp1=temp1+math.pow(dataobject.getYdata()[i]-meanY,2)
            temp2=temp2+math.pow(dataobject.getYdata()[i]-regression_function(dataobject.getXdata()[i]),2)
            
        #print("R^2:"+str(1-(temp2/temp1)))
        return 1-(temp2/temp1)

        
    @staticmethod
    def k_fold_validation(dataobject):
        folds=2
        trainrange=len(dataobject.getXdata())/folds #rundet automatisch nach unten
        testrange=int(trainrange+1)
        testobject=dc.data_class()
        trainobject=dc.data_class()
        testobject.changeYdata(dataobject.getYdata())
        sampleX=random.sample(list(enumerate(dataobject.getXdata())),testrange)
        testobject.changeXdata([sampleX[i][1] for i in range(testrange)])
        testobject.changeYdata([dataobject.getYdata()[sampleX[i][0]] for i in range(testrange)])       
        trainobject.changeXdata([dataobject.getXdata()[i] for i in range(len(dataobject.getXdata())) if i not in [sampleX[k][0] for k in range(len(sampleX))]])
        trainobject.changeYdata([dataobject.getYdata()[i] for i in range(len(dataobject.getXdata())) if i not in [sampleX[k][0] for k in range(len(sampleX))]])
        
        return {0: trainobject, 1:testobject}
        
        
        
           
    
    @staticmethod
    def cross_validation_type(number):
        cross_validation_type={0:linear_regression_algorithms.k_fold_validation}
        return cross_validation_type[number]
