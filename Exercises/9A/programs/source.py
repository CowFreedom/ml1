import numpy as np
import classification_algorithms as ca
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
def runClassification():
	#Loading data
	path="D:\\Documents\\Uni\\Programming\\Machine Learning Tutorium\\github Ordner\\Exercises\\9A\\data\\"
	trainX=np.loadtxt(path+"trainX.txt",encoding='latin1',delimiter=",")
	traint=np.loadtxt(path+"traint_numeric.txt",encoding='latin1',delimiter=",",dtype=int)
	#Building design matrix

	X=ca.buildLinearDesignMatrix(trainX)
	#calculation weights
	model=ca.multi_logistic_regression(trainX,traint,3)
	
	for i in range(len(trainX)):
		(assignment,prob)=model.predicted_label(trainX[i])
		#print("PROB: ",prob)
		if prob>1:
			#print("Probability greater 1!",prob)
			prob=1
		if (assignment==0):
			print("Klasse 1")
			col=color=(prob,0,0)
		elif (assignment==1):
			print("Klasse 2")
			col=color=(0,prob,0)
		elif (assignment==2):
			col=color=(0,0,prob)		
		
		
		if traint[i]==0:
			plt.plot(trainX[i][2],trainX[i][3],marker='o', linestyle='', color=col, label='Klasse 0')
		elif traint[i]==1:
			plt.plot(trainX[i][2],trainX[i][3],marker='x', linestyle='', color=col, label='Klasse 0')
			
		elif traint[i]==2:
			plt.plot(trainX[i][2],trainX[i][3],marker='^', linestyle='', color=col, label='Klasse 0')			
	plt.show()
			
	
	
	
def runClassification_sklearn():
	#Loading data
	path="D:\\Documents\\Uni\\Programming\\Machine Learning Tutorium\\github Ordner\\Exercises\\9A\\data\\"
	trainX=np.loadtxt(path+"trainX.txt",encoding='latin1',delimiter=",")
	traint=np.loadtxt(path+"traint_numeric.txt",encoding='latin1',delimiter=",",dtype=int)
	#Building design matrix

	#X=ca.buildLinearDesignMatrix(trainX)
	X=trainX
	#calculation weights
	logreg = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')

	# Create an instance of Logistic Regression Classifier and fit the data.
	logreg.fit(X,traint)
	
	classes=logreg.predict_proba(X)
	for i in range(len(trainX)):
		(assignment,prob)=(np.argmax(classes[i]),max(classes[i]))
		#print("PROB: ",prob)
		if prob>1:
			#print("Probability greater 1!",prob)
			prob=1
		if (assignment==0):
			print("Klasse 1")
			col=color=(prob,0,0)
		elif (assignment==1):
			print("Klasse 2")
			col=color=(0,prob,0)
		elif (assignment==2):
			col=color=(0,0,prob)		
		
		
		if traint[i]==0:
			plt.plot(trainX[i][2],trainX[i][3],marker='o', linestyle='', color=col, label='Klasse 0')
		elif traint[i]==1:
			plt.plot(trainX[i][2],trainX[i][3],marker='x', linestyle='', color=col, label='Klasse 0')
			
		elif traint[i]==2:
			plt.plot(trainX[i][2],trainX[i][3],marker='^', linestyle='', color=col, label='Klasse 0')	
			
	plt.xlabel('petal length')
	plt.ylabel('petal width')			
	'''
	# Plot also the training points
	plt.scatter(X[:, 2], X[:, 3], c=traint, edgecolors='k', cmap=plt.cm.Paired)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	print(X[:, 0],)
	'''
	plt.show()
	
def runClassification_sklearn_2classes():
	#Loading data
	path="D:\\Documents\\Uni\\Programming\\Machine Learning Tutorium\\github Ordner\\Exercises\\9A\\data\\"
	trainX=np.loadtxt(path+"trainX.txt",encoding='latin1',delimiter=",")
	traint=np.loadtxt(path+"traint_numeric.txt",encoding='latin1',delimiter=",",dtype=int)
	#adjust the data
	#If 1 then it is true
	traint_new0=np.array([ 0 if i!=0 else 1 for i in traint]) #Class 0 vs Rest
	traint_new1=np.array([ 0 if i!=1 else 1 for i in traint]) #Class 1 vs Rest
	traint_new2=np.array([ 0 if i!=2 else 1 for i in traint]) #Class 1 vs Rest
	
	#X=ca.buildLinearDesignMatrix(trainX)
	X=trainX
	#calculation weights
	logreg0 = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')
	logreg1 = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')
	logreg2 = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')

	# Create an instance of Logistic Regression Classifier and fit the data.
	logreg0.fit(X,traint_new0)
	logreg1.fit(X,traint_new1)
	logreg2.fit(X,traint_new2)	
	
	classes0=logreg0.predict_proba(X)
	classes1=logreg1.predict_proba(X)
	classes2=logreg2.predict_proba(X)

	
	for i in range(len(trainX)):
		prob0=classes0[i][1]
		prob1=classes1[i][1]
		prob2=classes2[i][1]
		print("PROB: ",prob0)

		if (prob0>prob1 and prob0 >prob2):
			print("Klasse 1")
			col=color=(prob0,0,0)
		elif (prob1>prob0 and prob1 >prob2):
			print("Klasse 2")
			col=color=(0,prob1,0)
		elif (prob2>prob0 and prob2 >prob1):
			col=color=(0,0,prob2)		
		
		
		if traint[i]==0:
			plt.plot(trainX[i][2],trainX[i][3],marker='o', linestyle='', color=col, label='Klasse 0')
		elif traint[i]==1:
			plt.plot(trainX[i][2],trainX[i][3],marker='x', linestyle='', color=col, label='Klasse 0')
			
		elif traint[i]==2:
			plt.plot(trainX[i][2],trainX[i][3],marker='^', linestyle='', color=col, label='Klasse 0')	
			
	plt.xlabel('petal length')
	plt.ylabel('petal width')			

	plt.show()

def runClassification_sklearn_artificial_data():
	#Loading data
	path="D:\\Documents\\Uni\\Programming\\Machine Learning Tutorium\\github Ordner\\Exercises\\9A\\data\\"
	trainX=np.loadtxt(path+"x_artificial.txt",encoding='latin1',delimiter=",")
	traint=np.loadtxt(path+"t_artificial.txt",encoding='latin1',delimiter=",",dtype=int)
	#Building design matrix

	#X=ca.buildLinearDesignMatrix(trainX)
	X=trainX
	#calculation weights
	logreg = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')

	# Create an instance of Logistic Regression Classifier and fit the data.
	logreg.fit(X,traint)
	
	classes=logreg.predict_proba(X)
	for i in range(len(trainX)):
		(assignment,prob)=(np.argmax(classes[i]),max(classes[i]))
		#print("PROB: ",prob)
		if prob>1:
			#print("Probability greater 1!",prob)
			prob=1
		if (assignment==0):
			print("Klasse 1")
			col=color=(prob,0,0)
		elif (assignment==1):
			print("Klasse 2")
			col=color=(0,prob,0)
		elif (assignment==2):
			col=color=(0,0,prob)		
		
		
		if traint[i]==0:
			plt.plot(trainX[i][2],trainX[i][3],marker='o', linestyle='', color=col, label='Klasse 0')
		elif traint[i]==1:
			plt.plot(trainX[i][2],trainX[i][3],marker='x', linestyle='', color=col, label='Klasse 0')
			
		elif traint[i]==2:
			plt.plot(trainX[i][2],trainX[i][3],marker='^', linestyle='', color=col, label='Klasse 0')	
			
	plt.xlabel('x1')
	plt.ylabel('x2')			
	'''
	# Plot also the training points
	plt.scatter(X[:, 2], X[:, 3], c=traint, edgecolors='k', cmap=plt.cm.Paired)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	print(X[:, 0],)
	'''
	plt.show()
	
def runClassification_sklearn_2classes_artificial_data():
	#Loading data
	path="D:\\Documents\\Uni\\Programming\\Machine Learning Tutorium\\github Ordner\\Exercises\\9A\\data\\"
	trainX=np.loadtxt(path+"x_artificial.txt",encoding='latin1',delimiter=",")
	traint=np.loadtxt(path+"t_artificial.txt",encoding='latin1',delimiter=",",dtype=int)
	#adjust the data
	#If 1 then it is true
	traint_new0=np.array([ 0 if i!=0 else 1 for i in traint]) #Class 0 vs Rest
	traint_new1=np.array([ 0 if i!=1 else 1 for i in traint]) #Class 1 vs Rest
	traint_new2=np.array([ 0 if i!=2 else 1 for i in traint]) #Class 1 vs Rest
	
	#X=ca.buildLinearDesignMatrix(trainX)
	X=trainX
	#calculation weights
	logreg0 = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')
	logreg1 = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')
	logreg2 = LogisticRegression(C=1e10, solver='lbfgs', multi_class='multinomial')

	# Create an instance of Logistic Regression Classifier and fit the data.
	logreg0.fit(X,traint_new0)
	logreg1.fit(X,traint_new1)
	logreg2.fit(X,traint_new2)	
	
	classes0=logreg0.predict_proba(X)
	classes1=logreg1.predict_proba(X)
	classes2=logreg2.predict_proba(X)

	
	for i in range(len(trainX)):
		prob0=classes0[i][1]
		prob1=classes1[i][1]
		prob2=classes2[i][1]
		print("PROB: ",prob0)

		if (prob0>prob1 and prob0 >prob2):
			print("Klasse 1")
			col=color=(prob0,0,0)
		elif (prob1>prob0 and prob1 >prob2):
			print("Klasse 2")
			col=color=(0,prob1,0)
		elif (prob2>prob0 and prob2 >prob1):
			col=color=(0,0,prob2)		
		
		
		if traint[i]==0:
			plt.plot(trainX[i][2],trainX[i][3],marker='o', linestyle='', color=col, label='Klasse 0')
		elif traint[i]==1:
			plt.plot(trainX[i][2],trainX[i][3],marker='x', linestyle='', color=col, label='Klasse 0')
			
		elif traint[i]==2:
			plt.plot(trainX[i][2],trainX[i][3],marker='^', linestyle='', color=col, label='Klasse 0')	
			
	plt.xlabel('x1')
	plt.ylabel('x2')			

	plt.show()	
	
	
def createArtificialData():
	c1=np.random.multivariate_normal([0,0,0,0],0.5* np.eye(4), 50)
	c2=np.random.multivariate_normal([1,1,1,1],0.5* np.eye(4), 50)
	c3=np.random.multivariate_normal([-1,-1,-1,-1],0.5* np.eye(4), 50)

	t1=0*np.ones(50)
	t2=1*np.ones(50)
	t3=2*np.ones(50)


	t=np.hstack((t1,t2,t3))
	X=np.ones(shape=(150,4))

	for i in range(50):
		X[i]=c1[i]

	for i in range(50,100):
		X[i]=c2[int(i%50)]

	for i in range(100,150):
		X[i]=c3[int(i%100)]

	path="D:\\Documents\\Uni\\Programming\\Machine Learning Tutorium\\github Ordner\\Exercises\\9A\\data\\"

	np.savetxt(path+'t_artificial.txt', t, delimiter='\n')
	np.savetxt(path+'x_artificial.txt', X, delimiter=',')

	plt.plot(c1[:,0],c1[:,1],linestyle=" ",marker="o")
	plt.plot(c2[:,0],c2[:,1],linestyle=" ",marker="o")
	plt.plot(c3[:,0],c3[:,1],linestyle=" ",marker="o")
	plt.show()


#runClassification_sklearn()
runClassification_sklearn_artificial_data()
runClassification_sklearn_2classes_artificial_data()