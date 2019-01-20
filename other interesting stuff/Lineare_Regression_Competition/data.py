# -*- coding: utf-8 -*-
import numpy as np

class data_class(object):
    
    x_dimension=0
    y_dimension=0

    x_data=[]
    y_data=[]
    #Funktioniert nur für ein- oder zweidimensionale Listen bzw. Numpy Arrays
    def determine_dimension(self, input):
        current_dimension=0
        if(isinstance(input[0],(list,np.ndarray))==True):
            current_dimension=len(input[0])
        elif isinstance(input[0],( int, float ))==True:
            current_dimension=1        
        else:
            current_dimension=float('nan')
            
        return current_dimension
  
    def changeXdata(self, input):
        self.x_dimension=self.determine_dimension(input)
        self.x_data=input
    
    def changeYdata(self, input):
        self.y_dimension=self.determine_dimension(input)
        self.y_data=input
        
    def getXdimension(self):
        return self.x_dimension
        
    def getYdimension(self):
        return self.y_dimension
    
    def getXdata(self):
        return self.x_data
        
    def getYdata(self):
        return self.y_data
                  
        
    def __init__(self):
          return   
