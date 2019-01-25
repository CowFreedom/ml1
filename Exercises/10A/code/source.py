import ctypes
import numpy.ctypeslib as ctl
import numpy as np

_sum = ctypes.CDLL('D:\\Documents\\Uni\\Programming\\Machine Learning Tutorium\\github Ordner\\Exercises\\10A\\code\\C++\\newton_banana.so')
#_sum.our_function.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_sum.newton_banana.argtypes = [ctl.ndpointer(np.float64,flags='aligned, c_contiguous')]


def newton_banana(x_old):
    global _sum
    #result = _sum.our_function(ctypes.c_int(num_numbers), array_type(*numbers))
    #result = _sum.newton_banana(x_old)
    _sum.newton_banana(x_old)
    
	
x_0=np.array([1,-0.05], dtype=np.float64)
newton_banana(x_0)
print(x_0)
'''	
	
def our_function2(numbers):
	sum=0
	for x in numbers:
		sum+=x
	return sum

def test_ourfunction1(n):
		x=np.random.randint(1,3,n, dtype=np.int)
		our_function(x)
		
def test_ourfunction2(n):
		x=np.random.randint(1,3,n)
		our_function2(x)
		
def test_ourfunction3(n):
		x=np.random.randint(1,3,n)
		return np.sum(x)	
		
	
from timeit import timeit
setup = 'from __main__ import test_ourfunction1,test_ourfunction2,test_ourfunction3, n ; import numpy as np'
num = 4#set num=40 for comparison with C++
n= 1000000000#set n=1000000 for comparison with C++
t1 = timeit('test_ourfunction1(n)', setup=setup, number=num)
t2 = timeit('test_ourfunction2(n)', setup=setup, number=num)
t3 = timeit('test_ourfunction3(n)', setup=setup, number=num)
print('our_function C++ binding: {:0.1f} \n our_function Python: {:0.1f} \n our_function numpy: {:0.1f}'.format(t1,t2,t3))

'''
