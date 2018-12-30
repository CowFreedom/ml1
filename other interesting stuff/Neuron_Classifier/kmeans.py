import numpy as np

class kmeans:
	def __init__(self,data,numberClusters):
		assert len(data)>=numberClusters, "You want more centers than you have data."
		assert len(data)>0 and numberClusters>0, "You need more than 0 centers or data."		
		self.X=data
		self.k=numberClusters
		self.centers=self.X[np.random.choice(self.X.shape[0], self.k, replace=False), :] #Randomly select centers from data
		
	def run(self):
		iter=0
		#tempCenters=self.centers[:,None]
		
		for u in range(50):
			self.labels=np.argmin(np.linalg.norm(self.X-self.centers[:,None],axis=2),axis=0)
			result=np.zeros(shape=(self.k,len(self.X[0])))
			self.count=np.zeros(shape=(self.k,1))
			for index,x in enumerate(self.X):
				result[self.labels[index],:]+=self.X[index]
				self.count[self.labels[index]]+=1
			self.centers=result/np.array([k if k!= 0 else 1 for k in self.count])
			
	def run2(self):
		iter=0
		#tempCenters=self.centers[:,None]
		
		for u in range(300):
			self.labels=np.argmin(np.linalg.norm(self.X-self.centers[:,None],axis=2),axis=0)
			result=np.zeros(shape=(self.k,2))
			self.count=np.zeros(shape=(self.k,1))
			
			for i in range(self.k):
				self.centers[i]=self.X[self.labels==i].mean(0)

#X=data, k=number of clusters
def kmeans2(X,k):
		assert len(X)>=k, "You want more centers than you have data."
		assert len(X)>0 and k>0, "You need more than 0 centers or data."	
		iter=0
		
		centers=X[np.random.choice(self.X.shape[0], self.k, replace=False), :]		
		for u in range(50):
			labels=np.argmin(np.linalg.norm(X-centers[:,None],axis=2),axis=0)
			result=np.zeros(shape=(self.k,2))
			self.count=np.zeros(shape=(self.k,1))
			
			for i in range(self.k):
				centers[i]=X[labels==i].mean(0)
		
		return (centers,labels)

				
def k_means(data, k, number_of_iterations):
    n = len(data)
    number_of_features = data.shape[1]
    # Pick random indices for the initial centroids.
    initial_indices = np.random.choice(range(n), k)
    # We keep the centroids as |features| x k matrix.
    means = data[initial_indices].T
    # To avoid loops, we repeat the data k times depthwise and compute the
    # distance from each point to each centroid in one step in a
    # n x |features| x k tensor.
    repeated_data = np.stack([data] * k, axis=-1)
    all_rows = np.arange(n)
    zero = np.zeros([1, 1, 2])
    for _ in range(number_of_iterations):
        # Broadcast means across the repeated data matrix, gives us a
        # n x k matrix of distances.
        distances = np.sum(np.square(repeated_data - means), axis=1)
        # Find the index of the smallest distance (closest cluster) for each
        # point.
        assignment = np.argmin(distances, axis=-1)
        # Again to avoid a loop, we'll create a sparse matrix with k slots for
        # each point and fill exactly the one slot that the point was assigned
        # to. Then we reduce across all points to give us the sum of points for
        # each cluster.
        sparse = np.zeros([n, k, number_of_features])
        sparse[all_rows, assignment] = data
        # To compute the correct mean, we need to know how many points are
        # assigned to each cluster (without a loop).
        counts = (sparse != zero).sum(axis=0)
        # Compute new assignments.
        means = sparse.sum(axis=0).T / counts.clip(min=1).T
    return means.T

		