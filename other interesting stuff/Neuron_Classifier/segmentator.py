try:
    from PIL import Image
    from PIL import ImageDraw
    import numpy as np
except ImportError:
    import Image
    import ImageDraw
    print("IMPORT ERROR")

import numpy as np
import kmeans as ml1




class segmentator:
	def __init__(self, data):
		self.raw=data
		self.XDim=self.raw.shape[0] #vertical length
		self.YDim=self.raw.shape[1] #horizontal length
		self.raw_transformed=np.ones(shape=(self.XDim,self.YDim,3))
		self.convolution_whole(7)
		self.segment_image(3)
		#self.saveTransformedRawImage(1)
		self.saveSegmentedImage(1)

	
	#x,y=coordinates of pixel to be convoluted, img=image, n=size of convolution window
	def convolution_pointwise(self,x,y,n):
		s=10
		for k in range(3):
			sum=0.0
			iter=0
			for i in range(n):
				for j in range(n):
					if x+i>=0 and y+j >=0 and x+i<self.XDim and y+j<self.YDim:
						#sum=sum+(self.raw[x+i,y+j,k])*np.exp(-0.5*s*(n**2+i**2))
						sum=sum+(self.raw[x+i,y+j,k])
						iter=iter+1
			sum=sum/iter
			#if (sum>255):
			#	sum=255
			self.raw_transformed[x,y,k]=sum
		
	def convolution_whole(self,n):
		for i in range(self.XDim):
			for j in range(self.YDim):
				self.convolution_pointwise(i,j,n)
			if i%10==0:
				print("Convolution percent done:",int((i/self.XDim)*100),"%")
	
	def segment_image(self,k):
		meaner=ml1.kmeans(self.raw.reshape(self.XDim*self.YDim,3),k)
		
		meaner.run()
		#print(meaner.labels.shape)
		self.segmented_image=meaner.labels.reshape(self.XDim,self.YDim) #potential error
		print(meaner.centers)
		#print("internet kmeans centers:",ml1.k_means(np.random.randint(25, size=(10, 4), dtype=np.uint8),k,50))
		'''
		iter1=0
		iter2=0
		iter3=0
		print(meaner.labels.shape)
		for i in range(self.XDim):
			for j in range(self.YDim):
				if (self.segmented_image[i,j]==0):
					iter1=iter1+1
				elif (self.segmented_image[i,j]==1):
					iter2=iter2+1
				elif (self.segmented_image[i,j]==2):
					iter3=iter3+1
				else:
					print("HIYAT",self.segmented_image[i,j])
		print(iter1,iter2,iter3)
		'''
		#print(self.raw.reshape(self.XDim*self.YDim,3)[meaner.labels==2])
	#Given a label, return the color
	def returnColor(self,v):
		if (v==0):
			return [254,0,0]
		elif (v==1):
			return [0,10,0]
		elif (v==2):
			return [0,0,250]
		elif (v==3):
			return [60,80,100]
		elif (v==4):
			return [254,10,200]
		elif (v==5):
			return [20,234,30]
		else:
			print("HELS",v)
			return [0,0,0]

	def saveTransformedRawImage(self,index):
		print("Saving image")
		transformedimage=np.zeros((self.XDim,self.YDim,3),dtype=np.uint8)
		for i in range(self.XDim):
			for j in range(self.YDim):
				transformedimage[i,j]=self.raw_transformed[i,j]
				#colorimage[i,j]=[255,255,255]
				#print(colorimage[i,j])		
		Image.fromarray(transformedimage, 'RGB').save('D://Documents//Uni//Programming//Machine Learning Tutorium//github Ordner//other interesting stuff//Neuron_Classifier//segmented_images//transformed_image'+str(index)+'.png',"PNG")
		
			
	def saveSegmentedImage(self,index):
		print("Saving image")
		colorimage=np.zeros((self.XDim,self.YDim,3),dtype=np.uint8)
		for i in range(self.XDim):
			for j in range(self.YDim):
				colorimage[i,j]=self.returnColor(self.segmented_image[i,j])
				#colorimage[i,j]=[255,255,255]
				#print(colorimage[i,j])
		
	
		Image.fromarray(colorimage, 'RGB').save('D://Documents//Uni//Programming//Machine Learning Tutorium//github Ordner//other interesting stuff//Neuron_Classifier//segmented_images//segmented_image'+str(index)+'.png',"PNG")
	
		
		
	
	
