#First, we load the data via stdin.
import sys
import os
import argparse
import numpy as np

#If no data is given via command line, then
#close the program
if len(sys.argv)==1:
	print("ML Submission Error: The test set has to be given as command-line argument")
	sys.exit()


parser = argparse.ArgumentParser()

parser.add_argument('infile',
                    default=sys.stdin,
                    type=argparse.FileType('r'),
                    nargs='?')

args = parser.parse_args()



data = np.fromstring(args.infile.read(), sep='\t') #this is only a one dimensional array, so we convert it to 2D below

yDim=int(len(data)/20)#should always be an integer, as joint size is multiple of 20

data=data.reshape(yDim,20)
#Now, we load the weights of our previous training (calculation of weights
#not part of the submission
#We assume the weights are stored in the path relative to where
#this script is executed 

dir_path = os.path.dirname(os.path.realpath(__file__))
print("DATA",data.shape)


class logistic_regression:
	def __init__(self,w):
		self.w=w

	def predicted_label(self,X):
		sigmoid=lambda x: 1/(1+np.exp(-x))
		likelihood=sigmoid(self.w.T.dot(X))
		if likelihood>=0.5:
			return 1
		else:
			return 0

path=dir_path+"\\weights.txt"

#Create the model
model=logistic_regression(np.loadtxt(path))

#Create the design matrix. Important: the dimension of the design matrix
#has to be the same as in the training phase. The basis function should be
#equal as well (e.g. polynomial basis in training as well as test evaluation)

def buildLinearDesignMatrix(data):
	X=np.ones(shape=(len(data),len(data[0])+1))
	
	for i in range(len(data)):
		X[i,1:,]=data[i]
	return X


	
X=buildLinearDesignMatrix(data)	#Create design matrix

#make predictions
predictions=np.zeros(shape=(len(data)))

for i in range(len(data)):
	predictions[i]=model.predicted_label(X[i])

#Printing to stdout. Has to be string the interpreter says	
sys.stdout.write(str(predictions))

