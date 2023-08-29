# How to import
import numpy as np

# Numpy array 1
cvalues = [20.1, 20.8, 21.9, 22.5, 22.7, 22.3, 21.8, 21.2, 20.9, 20.1]
print(cvalues)
# [20.1, 20.8, 21.9, 22.5, 22.7, 22.3, 21.8, 21.2, 20.9, 20.1]
C = np.array(cvalues)
print(type(C))
# <class 'numpy.ndarray'>
print(C)
# [20.1 20.8 21.9 22.5 22.7 22.3 21.8 21.2 20.9 20.1]


# Numpy array 2
print(C * 9 / 5 + 32)
# [68.18 69.44 71.42 72.5  72.86 72.14 71.24 70.16 69.62 68.18]

# With a Python list we would need to do the following:
print([x * 9 / 5 + 32 for x in cvalues])
# [68.18, 69.44, 71.42, 72.5, 72.86, 72.14, 71.24, 70.16, 69.62, 68.18]


# arange
import numpy as np

print(np.arange(10))
# [0 1 2 3 4 5 6 7 8 9]
print(np.arange(1, 10))
# [1 2 3 4 5 6 7 8 9]
print(range(1, 10))
# range(1, 10)
print(list(range(1, 10)))
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(np.arange(10.4))
# [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
print(np.arange(0.5, 10.4, 0.8))
# [ 0.5  1.3  2.1  2.9  3.7  4.5  5.3  6.1  6.9  7.7  8.5  9.3 10.1]
print(np.arange(0.5, 10.4, 0.8, int))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12]


# linespace 1
print(np.linspace(1, 10))
# [ 1.          1.18367347  1.36734694  1.55102041  1.73469388  1.91836735
#   2.10204082  2.28571429  2.46938776  2.65306122  2.83673469  3.02040816
#   3.20408163  3.3877551   3.57142857  3.75510204  3.93877551  4.12244898
#   4.30612245  4.48979592  4.67346939  4.85714286  5.04081633  5.2244898
#   5.40816327  5.59183673  5.7755102   5.95918367  6.14285714  6.32653061
#   6.51020408  6.69387755  6.87755102  7.06122449  7.24489796  7.42857143
#   7.6122449   7.79591837  7.97959184  8.16326531  8.34693878  8.53061224
#   8.71428571  8.89795918  9.08163265  9.26530612  9.44897959  9.63265306
#   9.81632653 10.        ]
print(np.linspace(1, 10, 7))
# [ 1.   2.5  4.   5.5  7.   8.5 10. ]
print(np.linspace(1, 10, 7, endpoint=False))
# [1.         2.28571429 3.57142857 4.85714286 6.14285714 7.42857143 8.71428571]


# linespace 2
samples, spacing = np.linspace(1, 10, retstep=True)
print(spacing, samples)
# 0.1836734693877551
# [ 1.          1.18367347  1.36734694  1.55102041  1.73469388  1.91836735
# ...
#   9.81632653 10.        ]
samples, spacing = np.linspace(1, 10, 20, endpoint=True, retstep=True)
print(spacing, samples)
# 0.47368421052631576
# [ 1.          1.47368421  1.94736842  2.42105263  2.89473684  3.36842105
# ...
#   9.52631579 10.        ]
samples, spacing = np.linspace(1, 10, 20, endpoint=False, retstep=True)
print(spacing, samples)
# 0.45
# [1.   1.45 1.9  2.35 2.8  3.25 3.7  4.15 4.6  5.05 5.5  5.95 6.4  6.85
#  7.3  7.75 8.2  8.65 9.1  9.55]


# Zero-dimensional Arrays in Numpy
x = np.array(42)
print("x: ", x)
print("The type of x: ", type(x))
print("The dimension of x:", np.ndim(x))
# x:  42
# The type of x:  <class 'numpy.ndarray'>
# The dimension of x: 0


# One-dimensional Arrays
I = np.array([1, 1, 2, 3, 5, 8, 13, 21])
F = np.array([3.4, 6.9, 99.8, 12.8])
print(I)  # [ 1  1  2  3  5  8 13 21]
print(F)  # [ 3.4  6.9 99.8 12.8]
print(I.dtype)  # int64
print(F.dtype)  # float64
print(np.ndim(I))  # 1
print(np.ndim(F))  # 1

# Two- and Multidimensional Arrays
A = np.array([[3.4, 8.7, 9.9],
              [1.1, -7.8, -0.7],
              [4.1, 12.3, 4.8]])
print(A)
# [[  3.4   8.7   9.9]
#  [  1.1  -7.8  -0.7]
#  [  4.1  12.3   4.8]]
print(A.ndim)
# 2

B = np.array([[[111, 112], [121, 122]],
              [[211, 212], [221, 222]],
              [[311, 312], [321, 322]]])
print(B)
# [[[111 112]
#   [121 122]]
#  [[211 212]
#   [221 222]]
#  [[311 312]
#   [321 322]]]
print(B.ndim)
# 3


# Shape of an Array
x = np.array([[67, 63, 87],
              [77, 69, 59],
              [85, 87, 99],
              [79, 72, 71],
              [63, 89, 93],
              [68, 92, 78]])
print(np.shape(x))  # (6, 3)
print(x.shape)  # (6, 3)

# Change the shape
x.shape = (3, 6)
print(x)
# [[67 63 87 77 69 59]
#  [85 87 99 79 72 71]
#  [63 89 93 68 92 78]]
x.shape = (2, 9)
print(x)
# [[67 63 87 77 69 59 85 87 99]
#  [79 72 71 63 89 93 68 92 78]]
y = x.reshape(3, 6)
print(y)
# [[67 63 87 77 69 59]
#  [85 87 99 79 72 71]
#  [63 89 93 68 92 78]]

# Indexing and Slicing
A = np.array([[3.4, 8.7, 9.9],
              [1.1, -7.8, -0.7],
              [4.1, 12.3, 4.8]])
print(A[1][
          0])  # Highly inefficient: We create an intermediate array A[1] from which we access the element with the index 0.
print(A[1, 0])  # More efficient

# Indexing and Slicing 2
S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(S[2:5])  # [2 3 4]
print(S[:4])  # [0 1 2 3]
print(S[6:])  # [6 7 8 9]
print(S[:])  # [0 1 2 3 4 5 6 7 8 9]

# Indexing and Slicing 3
A = np.array([
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55]])
print(A[:3, 2:])
# [[13 14 15]
#  [23 24 25]
#  [33 34 35]]

print(A[3:, :])
# [[41 42 43 44 45]
#  [51 52 53 54 55]]

X = np.array([
    [0, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27]])
print(X[::2, ::3])
# [[ 0  3  6]
#  [14 17 20]]

print(X[::, ::3])
# [[ 0  3  6]
#  [ 7 10 13]
#  [14 17 20]
#  [21 24 27]]

print(X[:2])
# [[ 0  1  2  3  4  5  6]
#  [ 7  8  9 10 11 12 13]]

A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
S = A[2:6]
S[0] = 22
S[1] = 23
print(A)
# [ 0  1 22 23  4  5  6  7  8  9]

A = np.arange(12)
print(A)
# [ 0  1  2  3  4  5  6  7  8  9 10 11]
B = A.reshape(3, 4)
A[0] = 42
print(B)
# [[42  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(np.may_share_memory(A, B))
# True


# Creating Arrays with Ones and Zeros
print(np.ones((2, 3)))
# [[1. 1. 1.]
#  [1. 1. 1.]]
print(np.ones((3, 4), dtype=int))
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
print(np.zeros((2, 4)))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
x = np.array([2, 5, 18, 14, 4])
print(np.ones_like(x))
# [1 1 1 1 1]
print(np.zeros_like(x))
# [0 0 0 0 0]


# Copying arrays
x = np.ones(10)
y = x
y[1] = 10
print(x)
# [ 1. 10.  1.  1.  1.  1.  1.  1.  1.  1.]

x = np.ones(10)
y = x.copy()
y[1] = 10
print(x)
# [ 1. 10.  1.  1.  1.  1.  1.  1.  1.  1.]


# Identity
print(np.identity(4))
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]


# Using scalars
# Without numpy
lst = [2, 3, 7.9, 3.3, 6.9, 0.11, 10.3, 12.9]
res = [val + 2 for val in lst]
print(res)
# [4, 5, 9.9, 5.3, 8.9, 2.11, 12.3, 14.9]

# With numpy
lst = [2, 3, 7.9, 3.3, 6.9, 0.11, 10.3, 12.9]
v = np.array(lst)
v = v + 2
print(v)
# [ 4.    5.    9.9   5.3   8.9   2.11 12.3  14.9 ]

import functools
import time
def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapperTimer(*args, **kwargs):
        startTime = time.perf_counter()
        value = func(*args, **kwargs)
        endTime = time.perf_counter()
        runTime = endTime - startTime
        print(f"Finished {func.__name__!r} in {runTime:.4f} secs")
        return value

    return wrapperTimer

@timer
def withoutNumpy():
    lst = np.random.randint(0, 100, (1000, 1000))
    lst = [val + 2 for val in lst]

@timer
def withNumpy():
    lst = np.random.randint(0, 100, (1000, 1000))
    lst += 2

withoutNumpy()
withNumpy()
# Finished 'withoutNumpy' in 0.0176 secs
# Finished 'withNumpy' in 0.0082 secs

print(v * 2.2)
# [ 8.8   11.    21.78  11.66  19.58   4.642 27.06  32.78 ]
print(v - 1.38)
# [ 2.62  3.62  8.52  3.92  7.52  0.73 10.92 13.52]
print(v ** 2)
# [ 16.      25.      98.01    28.09    79.21     4.4521 151.29   222.01  ]


# Arithmetic Operations with two Arrays
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.ones((3,3))
print(A + B)
# [[12. 13. 14.]
#  [22. 23. 24.]
#  [32. 33. 34.]]
print(A * (B + 1))
# [[22. 24. 26.]
#  [42. 44. 46.]
#  [62. 64. 66.]]


# Matrix Multiplication
A = np.array([[1, 2, 3], [2, 2, 2], [3, 3, 3]])
B = np.array([[3, 2, 1], [1, 2, 3], [-1, -2, -3]])
print(np.dot(A, B))
# [[ 2  0 -2]
#  [ 6  4  2]
#  [ 9  6  3]]
print(A*B)
# [[ 3  4  3]
#  [ 2  4  6]
#  [-3 -6 -9]]
print(np.mat(A) * np.mat(B))
# [[ 2  0 -2]
#  [ 6  4  2]
#  [ 9  6  3]]


# Comparision operators
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([ [11, 102, 13], [201, 22, 203], [31, 32, 303] ])
print(A == B)
# [[ True False  True]
#  [False  True False]
#  [ True  True False]]

print(np.array_equal(A, B))
print(np.array_equal(A, A))
# False
# True


# Logical operators
a = np.array([ [True, True], [False, False]])
b = np.array([ [True, False], [True, False]])
print(np.logical_or(a, b))
# [[ True  True]
#  [ True False]]
print(np.logical_and(a, b))
# [[ True False]
#  [False False]]


# Flatten and Reshape Arrays
A = np.array([[[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7]],
              [[ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15]]])
print(A.flatten())
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
print(A.flatten(order="C"))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
print(A.flatten(order="F"))
# [ 0  8  2 10  4 12  6 14  1  9  3 11  5 13  7 15]

X = np.array(range(8))
print(X)
# [0 1 2 3 4 5 6 7]
print(X.reshape((4,2)))
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]


# Concatenating Arrays
x = np.array([11,22])
y = np.array([18,7,6])
z = np.array([1,3,5])
c = np.concatenate((x,y,z))
print(c)
# [11 22 18  7  6  1  3  5]


x = np.array(range(24))
x = x.reshape((3,4,2))
y = np.array(range(100,124))
y = y.reshape((3,4,2))
z = np.concatenate((x,y))
print(z)
# [[[  0   1]
#   [  2   3]
#   [  4   5]
#   [  6   7]]
#
#  [[  8   9]
#   [ 10  11]
#   [ 12  13]
#   [ 14  15]]
#
#  [[ 16  17]
#   [ 18  19]
#   [ 20  21]
#   [ 22  23]]
#
#  [[100 101]
#   [102 103]
#   [104 105]
#   [106 107]]
#
#  [[108 109]
#   [110 111]
#   [112 113]
#   [114 115]]
#
#  [[116 117]
#   [118 119]
#   [120 121]
#   [122 123]]]

x = np.array(range(24))
x = x.reshape((3,4,2))
y = np.array(range(100,124))
y = y.reshape((3,4,2))
z = np.concatenate((x,y),axis = 1)
print(z)
# [[[  0   1]
#   [  2   3]
#   [  4   5]
#   [  6   7]
#   [100 101]
#   [102 103]
#   [104 105]
#   [106 107]]
#
#  [[  8   9]
#   [ 10  11]
#   [ 12  13]
#   [ 14  15]
#   [108 109]
#   [110 111]
#   [112 113]
#   [114 115]]
#
#  [[ 16  17]
#   [ 18  19]
#   [ 20  21]
#   [ 22  23]
#   [116 117]
#   [118 119]
#   [120 121]
#   [122 123]]]


# Adding New Dimensions
x = np.array([2,5,18,14,4])
print(x)
# [ 2  5 18 14  4]
y = x[:, np.newaxis]
print(y)
# [[ 2]
#  [ 5]
#  [18]
#  [14]
#  [ 4]]

