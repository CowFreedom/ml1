import numpy as np

def dot_product_iterative(vec1,vec2):
	sum=0
	for i in range(len(vec1)):
		sum+=(vec1[i]*vec2[i])
	return sum

def dot_product_recursive(vec1,vec2, n):
	if n==1:
		return vec1[n-1]*vec2[n-1]
	else:
		return vec1[n-1]*vec2[n-1]+dot_product_recursive(vec1,vec2,n-1)
		

def dot_product_numpy(vec1,vec2):
	return np.dot(vec1,vec2)
	
	
def test_dot_product_iterative(n):
	x=np.random.randint(20,size=n)
	y=np.random.randint(20,size=n)
	return dot_product_iterative(x,y)

def test_dot_product_recursive(n):
	x=np.random.randint(20,size=n)
	y=np.random.randint(20,size=n)
	return dot_product_recursive(x,y,n)		

def test_dot_product_numpy(n):
	x=np.random.randint(20,size=n)
	y=np.random.randint(20,size=n)
	return dot_product_numpy(x,y)	
	
from timeit import timeit
setup = 'from __main__ import test_dot_product_iterative,test_dot_product_recursive,test_dot_product_numpy, n ; import numpy as np'
num = 4000 #set num=40 for comparison with C++
n= 900#set n=1000000 for comparison with C++
t1 = timeit('test_dot_product_iterative(n)', setup=setup, number=num)
t2 = timeit('test_dot_product_recursive(n)', setup=setup, number=num)
t3 = timeit('test_dot_product_numpy(n)', setup=setup, number=num)
print('dot_product_iterative: {:0.1f} \n dot_product_recursive: {:0.1f} \n dot_product_numpy: {:0.1f}'.format(t1,t2,t3))
		
