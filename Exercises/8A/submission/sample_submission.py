#First, we load the data via stdin.
#See here: https://stackoverflow.com/questions/1450393/how-do-you-read-from-stdin


import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('infile',
                    default=sys.stdin,
                    type=argparse.FileType('r'),
                    nargs='?')

args = parser.parse_args()

#data = args.infile.read()

#Now, we load the weights of our previous training (calculation of weights
#not part of the submission
#We assume the weights are stored in the path relative to where
#this script is executed 
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

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


#NOT FINISHED MORE TODO