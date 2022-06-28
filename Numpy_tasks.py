#!/usr/bin/env python
# coding: utf-8

# # Numpy
# 
# 

# #### 1. Import the numpy package under the name `np` (★☆☆) 
# (**hint**: import … as …)

# In[27]:


import numpy as np


# #### 2. Print the numpy version and the configuration (★☆☆) 
# (**hint**: np.\_\_version\_\_, np.show\_config)

# In[28]:


print(np.__version__)
print(np.show_config())   


# #### 3. Create a null vector of size 10 (★☆☆) 
# (**hint**: np.zeros)

# In[12]:


np.zeros(10)


# #### 4.  How to find the memory size of any array (★☆☆) 
# (**hint**: size, itemsize)

# In[7]:


a=np.arange(10)
memory_size = a.size * a.itemsize
print(memory_size)


# #### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆) 
# (**hint**: np.info)

# In[8]:


np.info(np.add)


# #### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 
# (**hint**: array\[4\])

# In[9]:


a = np.zeros(10)
a[4] = 1
print(a)


# #### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) 
# (**hint**: np.arange)

# In[10]:


a= np.arange(10,49)
a


# #### 8.  Reverse a vector (first element becomes last) (★☆☆) 
# (**hint**: array\[::-1\])

# In[11]:


a= [0,1,2,3,4,5]
b = a[::-1]
b


# #### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 
# (**hint**: reshape)

# In[12]:


a = np.arange(0,9).reshape(3,3)
a


# #### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 
# (**hint**: np.nonzero)

# In[13]:


a=[1,2,0,0,4,0]
b= np.nonzero(a)
b


# #### 11. Create a 3x3 identity matrix (★☆☆) 
# (**hint**: np.eye)

# In[14]:


np.eye(3,3)


# #### 12. Create a 3x3x3 array with random values (★☆☆) 
# (**hint**: np.random.random)

# In[15]:


from numpy import random
c = np.random.random((3,3,3))
c


# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 
# (**hint**: min, max)

# In[16]:


from numpy import random
c = np.random.random((10,10))
print(np.min(c))
print(np.max(c))


# #### 14. Create a random vector of size 30 and find the mean value (★☆☆) 
# (**hint**: mean)

# In[17]:


from numpy import random
c = np.random.random((30))
print(np.mean(c))


# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
# (**hint**: array\[1:-1, 1:-1\])

# In[18]:


x = np.ones((5,5))
print("Existing array:")
print(x)
print("Result")
x[1:-1,1:-1] = 0
print(x)


# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) 
# (**hint**: np.pad)

# In[19]:


x = np.ones((4,4))
x = np.pad(x, pad_width=1, mode='constant', constant_values=0)
x


# #### 17. What is the result of the following expression? (★☆☆) 
# (**hint**: NaN = not a number, inf = infinity)

# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# 0.3 == 3 * 0.1
# ```

# In[20]:


False


# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 
# (**hint**: np.diag)

# In[21]:


np.diag(np.arange(1,5), k =-1)


# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 
# (**hint**: array\[::2\])

# In[22]:


a = np.ones((8,8))
a[0::2,1::2]=0
a[1::2, 0::2]=0
a


# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 
# (**hint**: np.unravel_index)

# In[30]:


print (np.unravel_index(100, (6,7,8)))


# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) 
# (**hint**: np.tile)

# In[30]:


a= np.array([[1,0],[0,1]])
b= np.tile(a,(4,4))
b


# #### 22. Normalize a 5x5 random matrix (★☆☆) 
# (**hint**: (x - min) / (max - min))

# In[41]:


a= np.arange(0,25).reshape(5,5)
max = np.max(a)
min = np.min(a)
diff = np.max(a) - np.min(a)
normalization = (a-min)/diff
normalization


# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) 
# (**hint**: np.dtype)

# In[53]:


RGBA = np.dtype([('red',np.uint8),('green',np.uint8),('blue',np.uint8),('alpha',np.uint8)])
arr = np.array((1,2,4,3),dtype = RGBA)
#print(arr['red'], arr['green'],arr ['blue'], arr['alpha'])
print(arr)


# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 
# (**hint**: np.dot | @)

# In[54]:


a = np.arange(0,15).reshape(5,3)
b = np.arange(0,6).reshape(3,2)
np.dot(a,b)


# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) 
# (**hint**: >, <=)

# In[59]:


a= np.array([1,2,3,8,6,9,7,5,8,6,6,89,6,9,6,6])
b = a [(a>3) & (a<=8 )]
b*= -1
b


# #### 26. What is the output of the following script? (★☆☆) 
# (**hint**: np.sum)

# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

# In[62]:


10
10


# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

# In[63]:


all expressions are legal


# #### 28. What are the result of the following expressions?

# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

# In[4]:


array([-2.14748365e+09])


# #### 29. How to round away from zero a float array ? (★☆☆) 
# (**hint**: np.uniform, np.copysign, np.ceil, np.abs)

# In[8]:


a = np.random.uniform(-10,+10,10)

print (np.copysign(np.ceil(np.abs(a)), a))


# #### 30. How to find common values between two arrays? (★☆☆) 
# (**hint**: np.intersect1d)

# In[9]:


a = [0,1,2,3,4,5,6]
b = [3,4,5,6,45,78,10]
np.intersect1d (a,b)


# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) 
# (**hint**: np.seterr, np.errstate)

# In[22]:


a = np.seterr (all = "ignore")
x_error = np.ones(1)/0
c = np.seterr(**a)


# #### 32. Is the following expressions true? (★☆☆) 
# (**hint**: imaginary number)

# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

# In[23]:


False


# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) 
# (**hint**: np.datetime64, np.timedelta64)

# In[24]:


yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday,today,tomorrow)


# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) 
# (**hint**: np.arange(dtype=datetime64\['D'\]))

# In[30]:


july_data = np.arange('2016-07', '2016-08', dtype=  'datetime64[D]')
july_data
#np.arange('2016-07', '2016-08', dtype='datetime64[D]')


# #### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) 
# (**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))

# In[38]:


A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)


# #### 36. Extract the integer part of a random array using 5 different methods (★★☆) 
# (**hint**: %, np.floor, np.ceil, astype, np.trunc)

# In[50]:


a = np.random.random(15)
print (a-a%1)
b=np.floor(a)
print(b)
c=np.ceil(a)
print(c)
d= a.astype(int)
print(d)
e= np.trunc(a)
print (e)


# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 
# (**hint**: np.arange)

# In[56]:


Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)


# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) 
# (**hint**: np.fromiter)

# In[21]:


generator = (x*x for x in range(10))
print(generator)
np.fromiter(generator, int, count=10)


# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 
# (**hint**: np.linspace)

# In[24]:


np.linspace(0,1,12,endpoint=True)[1:-1]


# #### 40. Create a random vector of size 10 and sort it (★★☆) 
# (**hint**: sort)

# In[33]:


a= np.random.random(10)
#a = np.arange(0,10)
a
np.sort(a)


# #### 41. How to sum a small array faster than np.sum? (★★☆) 
# (**hint**: np.add.reduce)

# In[35]:


np.add.reduce(np.arange(4))


# #### 42. Consider two random array A and B, check if they are equal (★★☆) 
# (**hint**: np.allclose, np.array\_equal)

# In[39]:


a= np.random.random(12)
b = np.random.random(12)
c= np.array_equal(a,b)
#d= np.allclose(a,b)
print (c)


# #### 43. Make an array immutable (read-only) (★★☆) 
# (**hint**: flags.writeable)

# In[43]:


arr = np.arange(10)
arr. flags.writeable = False
#arr


# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) 
# (**hint**: np.sqrt, np.arctan2)

# In[49]:


array_1 = np.random.random(20).reshape(10,2)
array_2 = np.random.random(20).reshape(10,2)
np.arctan2(array_1, array_2)
polar =  np.sqrt(array_1**2 + array_2**2)
polar


# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 
# (**hint**: argmax)

# In[4]:


x = np.arange(10)
a = np.argmax(x)
x[a] = 0
print(x)


# 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) 
# (**hint**: np.meshgrid)

# In[10]:


x,y = (4,5)
mx = np.linspace(0,1, x)
my = np.linspace (0,1, y)
np.meshgrid(mx, my)


# ####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) 
# (**hint**: np.subtract.outer)

# In[13]:


x = np.arange(0,6)
y = np.arange (10,15)
sub_array = np.subtract.outer(x,y)
1/sub_array


# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 
# (**hint**: np.iinfo, np.finfo, eps)

# In[16]:


for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)


# #### 49. How to print all the values of an array? (★★☆) 
# (**hint**: np.set\_printoptions)

# In[19]:


np.set_printoptions(threshold = np.inf)
array = np.arange(0,13)
array


# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) 
# (**hint**: argmin)

# In[24]:


array_1 = [10,34,56,78,0,4,5]
a = np.argmin(array_1)
print(a)


# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) 
# (**hint**: dtype)

# In[27]:


Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)


# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) 
# (**hint**: np.atleast\_2d, T, np.sqrt)

# In[41]:


Z = np.random.random((100,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)


# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? 
# (**hint**: astype(copy=False))

# In[49]:


array_1 = np.array([1.4, 1.7, 1.9 ,5.6,7.8,10])
array_1.astype(int, copy=True)


# #### 54. How to read the following file? (★★☆) 
# (**hint**: np.genfromtxt)

# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

# In[62]:


from io import StringIO
s = StringIO("1, 2, 3, 4, 5,6,  ,  , 7, 8,  , 9,10,11")
data = np.genfromtxt(s, dtype=np.int, delimiter=",")
data


# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) 
# (**hint**: np.ndenumerate, np.ndindex)

# In[65]:


for index in np.ndindex(1,4,1):
    print(index)
a = np.array([[1, 2], [3, 4]])
print()
for index, x in np.ndenumerate(a):
    print(index, x)


# #### 56. Generate a generic 2D Gaussian-like array (★★☆) 
# (**hint**: np.meshgrid, np.exp)

# In[67]:


X, Y = np.meshgrid(np.linspace(-1,1,2), np.linspace(-1,1,2))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
Gauss_array = np.exp(-( (D-mu)*2 / ( 2.0 * sigma*2 ) ) )
print(Gauss_array)


# #### 57. How to randomly place p elements in a 2D array? (★★☆) 
# (**hint**: np.put, np.random.choice)

# In[15]:


# a = np.arange(6)
# np.put(a,[0,4],[-34,-90])
# a

a = np.arange(6)
p=80  # proballity of replacing the number
np.put(a, np.random.choice(range(6),p,),30)
a


# #### 58. Subtract the mean of each row of a matrix (★★☆) 
# (**hint**: mean(axis=,keepdims=))

# In[23]:


array = np.arange(16).reshape(4,4)
print(array)
mean = array - np.mean(array, axis=1, keepdims= True)
print(mean)

#axis = 0 (rows)
#axis = 1 (columns)


# #### 59. How to sort an array by the nth column? (★★☆) 
# (**hint**: argsort)

# In[32]:


array = [4,67,8,9,56,78,90,56]
np.argsort(array)


# #### 60. How to tell if a given 2D array has null columns? (★★☆) 
# (**hint**: any, ~)

# In[38]:


arr_12 = np.arange(1,5).reshape(2,2)
(~arr_12.any(axis=0)).any()


# #### 61. Find the nearest value from a given value in an array (★★☆) 
# (**hint**: np.abs, argmin, flat)

# In[43]:


array = np.arange(5,15)
print(array)
np.argmin(array)


# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) 
# (**hint**: np.nditer)

# In[44]:


np.info(np.abs)


# #### 63. Create an array class that has a name attribute (★★☆) 
# (**hint**: class method)

# In[ ]:





# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) 
# (**hint**: np.bincount | np.add.at)

# In[ ]:





# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) 
# (**hint**: np.bincount)

# In[ ]:





# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) 
# (**hint**: np.unique)

# In[ ]:





# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) 
# (**hint**: sum(axis=(-2,-1)))

# In[ ]:





# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) 
# (**hint**: np.bincount)

# In[ ]:





# #### 69. How to get the diagonal of a dot product? (★★★) 
# (**hint**: np.diag)

# In[ ]:





# #### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 
# (**hint**: array\[::4\])

# In[ ]:





# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 
# (**hint**: array\[:, :, None\])

# In[ ]:





# #### 72. How to swap two rows of an array? (★★★) 
# (**hint**: array\[\[\]\] = array\[\[\]\])

# In[ ]:





# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) 
# (**hint**: repeat, np.roll, np.sort, view, np.unique)

# In[ ]:





# #### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) 
# (**hint**: np.repeat)

# In[ ]:





# #### 75. How to compute averages using a sliding window over an array? (★★★) 
# (**hint**: np.cumsum)

# In[ ]:





# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) 
# (**hint**: from numpy.lib import stride_tricks)

# In[ ]:





# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) 
# (**hint**: np.logical_not, np.negative)

# In[ ]:





# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

# In[ ]:





# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

# In[ ]:





# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) 
# (**hint**: minimum, maximum)

# In[ ]:





# #### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) 
# (**hint**: stride\_tricks.as\_strided)

# In[ ]:





# #### 82. Compute a matrix rank (★★★) 
# (**hint**: np.linalg.svd) (suggestion: np.linalg.svd)

# In[ ]:





# #### 83. How to find the most frequent value in an array? 
# (**hint**: np.bincount, argmax)

# In[ ]:





# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 
# (**hint**: stride\_tricks.as\_strided)

# In[ ]:





# #### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) 
# (**hint**: class method)

# In[ ]:





# #### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) 
# (**hint**: np.tensordot)

# In[ ]:





# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) 
# (**hint**: np.add.reduceat)

# In[ ]:





# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

# In[ ]:





# #### 89. How to get the n largest values of an array (★★★) 
# (**hint**: np.argsort | np.argpartition)

# In[ ]:





# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) 
# (**hint**: np.indices)

# In[ ]:





# #### 91. How to create a record array from a regular array? (★★★) 
# (**hint**: np.core.records.fromarrays)

# In[ ]:





# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) 
# (**hint**: np.power, \*, np.einsum)

# In[ ]:





# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) 
# (**hint**: np.where)

# In[ ]:





# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

# In[ ]:





# #### 95. Convert a vector of ints into a matrix binary representation (★★★) 
# (**hint**: np.unpackbits)

# In[ ]:





# #### 96. Given a two dimensional array, how to extract unique rows? (★★★) 
# (**hint**: np.ascontiguousarray)

# In[ ]:





# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) 
# (**hint**: np.einsum)

# In[ ]:





# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? 
# (**hint**: np.cumsum, np.interp)

# In[ ]:





# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) 
# (**hint**: np.logical\_and.reduce, np.mod)

# In[ ]:





# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) 
# (**hint**: np.percentile)

# In[ ]:




