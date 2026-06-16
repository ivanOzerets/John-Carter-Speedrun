Multiple Features
- instead of just price of houses, have other features per data point
- denoted x_1, x_2, x_3 …
- x_j = jth feature
- n =  number of features
- x^i is a vector of features

Parameters of the model
- w = [w_1 w_2 … w_n]
- b

Multiple linear regression (not multivariate)

Vectorization
- NumPy: numerical linear algebra library
- indexing from 0
- without vectorization → summation / code → for loop
- dot product is vectorized
- code is shorter and runs faster
- faster because numpy runs on parallel hardware

### Optional Lab: Python, NumPy and Vectorization

[Broadcasting — NumPy v2.3 Manual](https://numpy.org/doc/stable/user/basics.broadcasting.html)

Broadcasting
- broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations
- the smaller array is ‘broadcast’ across the larger array so that they have compatible shapes
- provides a means of vectorizing array operations so that looping occurs in C instead of Python
- no needless copies of data
- numpy operations are usually done on pairs of arrays on an element-by-element basis
- 2x2 x 2x2 being the simplest
```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])
a * b
>>> array([2.,  4.,  6.])
```
- constraints relax when the array shapes meet certain constraints
- simplest example occurs when an array and a scalar value are combined in an operation
```python
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b
>>> array([2.,  4.,  6.])
```
- the scalar b is being ‘stretched’ during the arithmetic operation into an array with the same shape as a
- the second example is more efficient than the first because broadcasting moves less memory around during the multiplication (b being a scalar)

General broadcasting rules
- numpy compares shapes of two arrays element-wise
- starts with the trailing (rightmost) dimension and works its way left
- dimensions are compatible when 
	- they are equal or
	- one of them is 1
- if these conditions are not met, a \<Value Error: operands could not be broadcast together\> exception is thrown, indicating incompatible shapes
- input array do not need to have the same number of dimensions
- resulting array will have the same number of dimensions as the input array with the greatest number of dimensions
- ‘size’ of each dimension is the largest size of the corresponding dimension among the input arrays
- missing dimensions are assumed to have size one
```python
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3
```
```python
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

Broadcastable arrays
- a set of array sis called ‘broadcastable’ to the same shape if the above rule produce a valid result
- if a.shape is (5,1), b.shape is (1,6), c.shape is (6,) and d.shape is () so that d is a scalar, than a, b, c, and d are all broadcastable to dimension (5,6) and 
	- a acts like a (5,6) array where a[:,0] is broadcast to the other columns
	- b acts like a (5,6) array where b[0,:] is broadcast to the other rows
	- c acts like a (1,6) array and therefore like a (5,6) array where c[:] is broadcast to every row
	- d acts like a (5,6) array where the single value is repeated
		- the colon in array slicing means all in the current dimension
		- the comma in array slicing means next dimension
- basically array version of scalar
```python
A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5
```
```python
a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
a[:, np.newaxis] + b # need to add np.newaxis to make a = (4,), I think
array([[ 1.,   2.,   3.],
       [11.,  12.,  13.],
       [21.,  22.,  23.],
       [31.,  32.,  33.]])
```
- newaxis index operator inserts a new axis into a, making it a two-dimensional 4x1 aray
- combining the 4x1 array with b, which has shape (3,), yields a 4x3 array

Practical example: vector quantization (VQ)
- VQ is used in information theory, classification, and other related areas
- the basic operation in VQ finds the closest point in a set of points, called codes, to a given point, called the observation
- in 2D case below, the values in observation describe the weight and height of an athlete to be classified
- the codes represent different classes of athletes
- finding the closet point requires calculating the distance between observation and each of the codes
- the shortest distance provides the best match
```python
observation = array([111.0, 188.0])
codes = array([[102.0, 203.0],
               [132.0, 193.0],
               [45.0, 155.0],
               [57.0, 173.0]])
diff = codes - observation    # the broadcast happens here
dist = sqrt(sum(diff**2,axis=-1))
argmin(dist)
>> 0
```
- when performing operations on array, the axis parameter often dictates along which dimension the operation should be applied
	- axis 0: in a 2d array, axis 0 typically represents the rows, and operations along axis 0 are performed column-wise
	- axis 1: row-wise
```python
Observation      (2d array):      10 x 3
Codes            (3d array):   5 x 1 x 3
Diff             (3d array):  5 x 10 x 3
```
- the three-dimensional array, diff, is a consequence of broadcasting, not a necessity for the calculation
- large data sets will generate a large intermediate array that is computationally inefficient
- instead, if each observation is calculated individually using a python loop around the code, a much smaller array is used
- ^ example of when not to use broadcasting
- uses unnecessarily large amounts of memory for a particular alg
- better to just use outer loop in python

Python and NumPy
- python has a set of numeric data types and arithmetic operations
- numpy is a library that extends the base capabilities of python to add a richer data set including more numeric types, vectors, matrices, and many matrix functions
- python arithmetic operators work on numpy data types and many numpy functions will accept python data types

Vectors
- Abstract
	- ordered arrays of numbers
	- denoted with lower case bold letter such as **x**
	- the elements of a vector are all the same type
	- the number of elements in an array is referred to as the dimension, in math → rank
	- when referencing individual elements will in be indicated with a subscript and not bold, x_0
	
- NumPy Arrays
	- basic structure, indexable, n-dimensional array containing elements of the same type (dtype)
	- dimension also refers to the number of indexes of an array
	- a 1d array has one index -. shape (n,): n elements indexed [0] through [n-1]
	
- Vector Creation
	- the first parameter is the shape of the object
	- this can either be a single value for a 1d result or a tuple (n,m,…) specifying the shape of the result 
	```python
# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
	```
	- arrays are initialized with floats / not true, see below example
	- (4) and ((4,)) are the same 
	```python
# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
	```
	- period after number means make output floats I think 
	```python
# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
	```
	- if one of the input data is a float, the rest will be

- Operations on Vectors
	- Indexing
		- elements of vectors can be accessed via indexing and slicing
		- indexing means referring to an element of an array by its position within the array
		- slicing means getting a subset of elements from an array based on their indices 
		```python
#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
    
# [0 1 2 3 4 5 6 7 8 9]
# a[2].shape: () a[2]  = 2, Accessing an element returns a scalar
# a[-1] = 9
# The error message you'll see is:
# index 10 is out of bounds for axis 0 with size 10
		```
	- Slicing
		- creates an array of indices using a set of three values (start:stop:step)
		- a subset of values is also valid 
		```python
#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step) / stop not including
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)

# a        = [0 1 2 3 4 5 6 7 8 9]
# a[2:7:1] =  [2 3 4 5 6]
# a[2:7:2] =  [2 4 6]
# a[3:]    =  [3 4 5 6 7 8 9]
# a[:3]    =  [0 1 2]
# a[:]     =  [0 1 2 3 4 5 6 7 8 9]
		```
	- Single vector operations
		```python
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
b = -a 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

# a             : [1 2 3 4]
# b = -a        : [-1 -2 -3 -4]
# b = np.sum(a) : 10
# b = np.mean(a): 2.5
# b = a**2      : [ 1  4  9 16]
		```

- Vector Vector element-wise operations
	- most of the numpy arithmetic, logical and comparison operations apply to vectors as well
	- for example: c_i = a_i + b_i
	```python
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# Binary operators work element wise: [0 0 6 8]
	```
	```python
#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# The error message you'll see is:
# operands could not be broadcast together with shapes (4,) (2,) 
	```

- Scalar Vector operations
	- vectors can be ‘scaled’ by scalar values
	- a scalar value is just a number
	- scalar multiplies all the elements of the vector 
	```python
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a 
print(f"b = 5 * a : {b}")

# b = 5 * a : [ 5 10 15 20]
	```

- Vector Vector dot product
	- dot product is the mainstay of linear algebra and numpy
	- operation used extensively
		- multiplies the values in two vectors element-wise and then sums the result
	- requires that the dimensions of the two vectors are the same
	```python
def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x
	```
	```python
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

# my_dot(a, b) = 24
	```
	- now with np.dot()
	```python
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# NumPy 1-D np.dot(a, b) = 24, np.dot(a, b).shape = () 
# NumPy 1-D np.dot(b, a) = 24, np.dot(a, b).shape = () 
	```

- The Need for Speed: vector vs for loop
	- numpy library improves speed memory efficiency 
	```python
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory

# np.dot(a, b) =  2501072.5817
# Vectorized version duration: 184.7713 ms 
# my_dot(a, b) =  2501072.5817
# loop version duration: 9718.5793 ms 
	```
	- numpy makes better use of available data parallelism in the underlying hardware
	- GPU’s and modern CPU’s implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel

- Vector Vector operations in Course 1
	- going forward, examples will be stored in an array, X_train of dimension (m,n) → a 2d array or matrix
	- w will be a 1d vector of shape (n,)
	- operations performed by looping through the examples, extracting each example to work on individually by indexing X → X[i]
	- X[i] return a value of shape (n,), a 1d vector → consequently, operations involving X[i] are often vector-vector 
	```python
# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

# X[1] has shape (1,)
# w has shape (1,)
# c has shape ()
	```

Matrices
- Abstract
	- two dimensional arrays
	- the elements of a matrix are all of the same type
	- matrices are denoted with a capitol, bold letter such as **X**
	- m is often the number of rows and n the number of columns
	- elements of a matrix can be referenced with a 2d index
	- first index is row, second is column

- NumPy Arrays
	- numpy’s basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype)
	- matrices have 2d index [m,n]
	- 2d matrices are used to hold training data
	- training data is m examples by n features creating an (m,n) array

- Matrix Creation
	- the same functions that created 1d vectors will create 2d or nd arrays
	- NumPy uses brackets for each dimension
	- when printing, one row per line 
	```python
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 

# a shape = (1, 5), a = [[0. 0. 0. 0. 0.]]
# a shape = (2, 1), a = [[0.]
#                        [0.]]
# a shape = (1, 1), a = [[0.44236513]]
	```
	- you can manually specify data
	- dimensions are specified with additional brackets 
	```python
# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

#  a shape = (3, 1), np.array: a = [[5]
#                                   [4]
#                                   [3]]
#  a shape = (3, 1), np.array: a = [[5]
#                                   [4]
#                                   [3]]
	```

- Operations on Matrices
	- Indexing
		- matrices include a second index
		- the two indexes describe [row, column]
		- access can either return an element or a row/column 
		```python
#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

# a.shape: (3, 2), 
# a= [[0 1]
#     [2 3]
#     [4 5]]

# a[2,0].shape:   (), a[2,0] = 4,     type(a[2,0]) = <class 'numpy.int64'> Accessing an element returns a scalar

# a[2].shape:   (2,), a[2]   = [4 5], type(a[2])   = <class 'numpy.ndarray'>
		```
		- accessing a matrix by just specifying the row will return a 1d vector
		- Reshape
			- a = np.arange(6).reshape(-1,2)
			- first creates a 1d vector of six elements, than reshaped into a 2d array
			- could have been written a = np.arange(6).reshape(3,2)
			- the -1 argument tells the routine to compute the number of rows given the size of the array and the number of columns
			
	- Slicing
		- creates an array of indices using a set of three values (start:stop:step)
		- a subset of values is also valid 
		```python
#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")

# a = 
# [[ 0  1  2  3  4  5  6  7  8  9]
#  [10 11 12 13 14 15 16 17 18 19]]
# a[0, 2:7:1] =  [2 3 4 5 6] ,  a[0, 2:7:1].shape = (5,) a 1-D array
# a[:, 2:7:1] = 
#  [[ 2  3  4  5  6]
#  [12 13 14 15 16]] ,  a[:, 2:7:1].shape = (2, 5) a 2-D array
# a[:,:] = 
#  [[ 0  1  2  3  4  5  6  7  8  9]
#  [10 11 12 13 14 15 16 17 18 19]] ,  a[:,:].shape = (2, 10)
# a[1,:] =  [10 11 12 13 14 15 16 17 18 19] ,  a[1,:].shape = (10,) a 1-D array
# a[1]   =  [10 11 12 13 14 15 16 17 18 19] ,  a[1].shape   = (10,) a 1-D array
		```
	
Gradient descent for multiple linear regression
- with w as a vector
- gradient descent alg with multiple features
- you find each w’s cost

Alternative to gradient descent (finding w and b)
- normal equation
	- only for linear regression
	- solve for w, b without iteration
- disadvantages
	- doesn’t generalize to other learning algorithms
	- slow when the number of features is large (\>10,000)
- normal equation method may be used in machine learning libraries that implement linear regression

### Optional Lab: Multiple Variable Linear Regression

```python
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
```

Notation
- a → scalar, non bold
- **a** → vector, bold
- **A** → matrix, bold capital
- **X** → training example matrix / X_train
- **y **→ training example targets / y_train
- **X**^i, y^i → ith training example / X[i],  y[i]
- m → number of training example / m
- n → number of features in each example / n
- **w** → parameter: weight / w
- b → parameter: bais / b
- f_**w**,b(**x**^i) = **w **. **x**^i + b→ the result of the  model evaluation at **x**^i parameterized by **w**, b

Note: size in sqft rather than 1000 sqft will cause an issue → learn later

Matrix X containing our examples
- examples stored in a numpy matrix x_train
- each row of the matrix represents on example
- when you have m training examples, and there are n features, **X** is a matrix with dimension (m,n)

Parameter vector w, b
- **w **is a vector with **n** elements
	- each element contains the parameter associated with one feature
	- b is a scalar parameter
	
Model Prediction with Multiple Variables
- the model’s prediction with multiple variables is given by the linear model: f_**w**,b(**x**) = **w** . **x** + b
- Single Prediction element by element
	- with on feature, prediction multiplied one feature value by one parameter and added a bias parameter
	- direct extension to multiple features would be to implement using loop over each element, performing the multiply with its parameter and then adding the bias parameter at the end
		```python
def predict_single_loop(x, w, b): 
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p
		```
		```python
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# x_vec shape (4,), x_vec value: [2104    5    1   45]
# f_wb shape (), prediction: 459.9999976194083
		```

Single Prediction, vector
- np.dot()
	```python
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    
	```
	```python
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# x_vec shape (4,), x_vec value: [2104    5    1   45]
# f_wb shape (), prediction: 459.99999761940825
	```

Compute Cost With Multiple Variables
- the equation for the cost function with multiple variables **J**(**w**, b) is:
	```python
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost
	```
	```python
# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

# Cost at optimal w : 1.5578904880036537e-12
	```

Gradient Descent With Multiple Variables
- GD for multiple variables
- n is the number of features, parameters w_j, b, are updated simultaneously
- m is the number of training examples in the data set
- f_**w**,b(**x**^i) is the model’s prediction, while y^i is the target value

- Compute Gradient with Multiple Variables
	- outer loop over all m example
		- cost function derivative wrt b can be computed directly and accumulated
		- in a second loop over all n features:
			- cost function derivative wrt w is computed for each w_j 
			```python
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i] # literally copying math
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j] # copying math here as well
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw
			```
			```python
#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# dj_db at initial w,b: -1.673925169143331e-06
# dj_dw at initial w,b: 
#  [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]
			```

	- Gradient Descent With Multiple Variables 
	```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost at every interval 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0: #if multiple of 100
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
	```
	```python
# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape # getting m but not n with _ as blank
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    
# Iteration    0: Cost  2529.46   
# Iteration  100: Cost   695.99   
# Iteration  200: Cost   694.92   
# Iteration  300: Cost   693.86   
# Iteration  400: Cost   692.81   
# Iteration  500: Cost   691.77   
# Iteration  600: Cost   690.73   
# Iteration  700: Cost   689.71   
# Iteration  800: Cost   688.70   
# Iteration  900: Cost   687.69   
# b,w found by gradient descent: -0.00,[ 0.2   0.   -0.01 -0.07] 
# prediction: 426.19, target value: 460
# prediction: 286.17, target value: 232
# prediction: 171.47, target value: 178
	```
	- The results are not inspiring
	- cost is still declining and our predictions are not very accurate

Feature Scaling
- if a feature has a large number associated with it, it is more likely that a good model will have relatively small associated weight

Feature size and parameter size
- small change to w_1 can have a large impact on the estimated price and the cost J because w_1 is multiplied by a very large number
- w_2 has a small impact because a large change is needed make a significant change in the predictions
- because the contours are so tall and skinny, GD may end up bouncing back and forth
- instead, scale the features
- comparable ranges of values to each other

Feature scaling
- take each original x_1 value and divide by the maximum of the range

Mean normalization
- rescale so both are centered around zero
- average = mean

Z-score normalization
- these are rescaling methods
- standard deviation sigma

Feature scaling
- aim for about -1 ≤ x_j ≤ 1 for each feature
- -3 ≤ x_j ≤ 3 → acceptable
- -0.3 ≤ x_j ≤ 0.3 → acceptable
- 0 ≤ x_1 ≤ 3 → okay, no rescaling
- -2 ≤ x_2 ≤ 0.5 → okay, no rescaling
- -100 ≤ x_3 ≤ 100 → too large / rescale
- -0.001 ≤ x_4 ≤ 0.001 → too small / rescale
- 98.6 ≤ x_5 ≤ 105 → too large / rescale
- almost never any harm to run feature scaling

Checking GD for Convergence
- make sure gradient sis working correctly
- objective: find parameters w and b that minimize the cost function J
- number of variations needed varies
- automatic convergence test
	- let epsilon be 10^-3
	- if J decrease by ≤ epsilon in one iteration, declare convergence (found parameters w, b to get close to global minimum)
	- often finding the right threshold epsilon is pretty difficult, often looking at the graph is better

Choosing the Learning Rate
- if the cost goes up and down → bug in code, learning rate is too large
- with a small enough alpha J should decrease on every iteration
- you can initially set alpha to be a very small number → if no decrease, then bug
- if the learning rate is too small, GD takes a lot more iterations to converge
- try values of alpha of : 0.001    0.01    0.1     1
- then increase 3 fold

### Optional Lab: Feature scaling and Learning Rate (Multi-variable)

```python
# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
```
```python
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()
```

- plotting each feature vs the target, price, provides some indication of which features have the strongest influence on price
- increasing size increases price
- bedrooms and floors don’t seem to have a strong impact
- newer houses have higher prices than older houses

GD with Multiple Variables

Learning Rate
- set initially to alpha = 9.9e^-7
- learning rate too high

- set to alpha = 9e^-7
- not too bad of a learning rate
- the cost is decreasing as it should
- w_0 is still oscillating around the minimum, but the cost is decreasing with every iteration rather than increasing
- dj_dw[0] changes sign with each iteration as w[0] jumps over the optimal value → will converge

- set to alpha = 1e-7
- cost is decreasing as it should
- w_0 is approaching the minimum without oscillations
- dj_20 is negative throughout the run

Feature Scaling
- three techniques
	- feature scaling → dividing each possible feature by its maximum value / (x - min)/(max - min)
	- mean normalization → (x - mean)/(max - min)
	- z-score normalization → (x - mean)/(std)

Z-score normalization
- after z-score normalization, all features will have a mean of 0 and a std of 1
- when normalizing the features, it is important to store the valies used for normalization - the mean value and the std used for the computations
- after leaning the parameters form the model, we often want to predict the prices of houses we have not seen before
- given a new x value, we must first normalize x using the mean and std that we had previously computed from the training set

Implementation
- normalization
	```python
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#check our work
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
	```
	```python
mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()
	```
- left: unnormalized → range of values or the variance of the size(sqrt) feature is much larger than that of age
- middle: the first step removes the mean or average value from each feature, this leaves the features that are centered around zero. its’ difficult to see the difference for the ‘age feature, but size(sqtr) is clearly around zero
- right: the second step divides by the standard deviation, this leaves both features centered at zero with a similar scale

Peak to Peak
- the peak to peak range of each column is reduced from a factor of thousands to a factor of 2-3 by normalization
	```python
# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# X_mu = [1.42e+03 2.72e+00 1.38e+00 3.84e+01], 
# X_sigma = [411.62   0.65   0.49  25.78]
# Peak to Peak range by column in Raw        X:[2.41e+03 4.00e+00 1.00e+00 9.50e+01]
# Peak to Peak range by column in Normalized X:[5.85 6.14 2.06 3.69]
	```
	```python
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")

plt.show()
	```
- the range of the  normalized data is centered around zero and roughly +/- 2
- the range is similar for each feature

Rerunning GD
- with normalized data
- the scaled results get very accurate results much faster
- the gradient of each parameter is tiny by the end of a short run
- learning rate of 0.1 is a good start for regression with normalized features
- plotting predictions vs target values
- the prediction is made using the normalized feature while the plot is shown using the original feature values
	```python
#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

# plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
	```
- the results look good
- with multiple features, we can no longer have a single plot showing results vs features
- when generating the plot, the normalized training set must also be normalized

Prediction
- the point of generating our model is to use it to predict housing prices that are not in the data set
- to test, normalize the data with the mean and std derived when the training data was normalized
	```python
# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

# [-0.53  0.43 -0.79  0.06]
# predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = $318709
	```

Cost contours
- another way to view feature scaling is in terms of the cost contours
- when feature scales do not match, the plot of cost vs parameters in a contour plot is asymmetric

Feature Engineering
- choosing the right features is critical
- area of land might be more predictive of the price → intuition
- transforming or combining original features

Polynomial Regression
- if the data looks like a nonlinear function would fit better, then you could guess exponents would fit better
- is there a way to not guess? → later in the course

Choice of features
- could use sqrt of x

### Optional Lab: Feature Engineering and Polynomial Regression

Feature Engineering and Polynomial Regression Overview
- out of the box, linear regression provides a means of building models of the form: f_w,b = w_0x_0 + …+ w_(n-1)x_(n-1) + b
- what if your features/data are non-linear or are combinations of features?
- for example, housing prices do not tend to be linear with living area but penalize very small or very large houses
- no amount of adjusting w, b will achieve a fit to a non-linear curve

Polynomial Features
- start with a quadratic y = 1 + x^2
- np.c_[..] is a numpy routine to concatenate along the column boundary
	```python
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()

# Iteration         0, Cost: 1.65756e+03
# Iteration       100, Cost: 6.94549e+02
# Iteration       200, Cost: 5.88475e+02
# Iteration       300, Cost: 5.26414e+02
# Iteration       400, Cost: 4.90103e+02
# Iteration       500, Cost: 4.68858e+02
# Iteration       600, Cost: 4.56428e+02
# Iteration       700, Cost: 4.49155e+02
# Iteration       800, Cost: 4.44900e+02
# Iteration       900, Cost: 4.42411e+02
# w,b found by gradient descent: w: [18.7], b: -52.0834
	```
- what is needed is something like y = w_0x_0^2 + b or a polynomial feature
- can modify the input data to engineer the needed features
- if you swap the original data with a version that squares the x value, then you can achieve y = w_0x_0^2 + b
	```python
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature
	```
	```python
X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# Iteration         0, Cost: 7.32922e+03
# Iteration      1000, Cost: 2.24844e-01
# Iteration      2000, Cost: 2.22795e-01
# Iteration      3000, Cost: 2.20764e-01
# Iteration      4000, Cost: 2.18752e-01
# Iteration      5000, Cost: 2.16758e-01
# Iteration      6000, Cost: 2.14782e-01
# Iteration      7000, Cost: 2.12824e-01
# Iteration      8000, Cost: 2.10884e-01
# Iteration      9000, Cost: 2.08962e-01
# w,b found by gradient descent: w: [1.], b: 0.0490
	```
- GD modifies out initial values of w, b to be (1, 0,049) or a model of y = 1 * x_0^2 + 0.049
- close to our target of y = 1 * x_0^2 + 1

Selecting Features
- one could add a variety of potential features to try and find the most useful
- what if we had instead tried: y = w_0x_0 + w_1x_1^2 + w_2x_2^3 + b?
	```python
# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
	```
	```python
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# Iteration         0, Cost: 1.14029e+03
# Iteration      1000, Cost: 3.28539e+02
# Iteration      2000, Cost: 2.80443e+02
# Iteration      3000, Cost: 2.39389e+02
# Iteration      4000, Cost: 2.04344e+02
# Iteration      5000, Cost: 1.74430e+02
# Iteration      6000, Cost: 1.48896e+02
# Iteration      7000, Cost: 1.27100e+02
# Iteration      8000, Cost: 1.08495e+02
# Iteration      9000, Cost: 9.26132e+01
# w,b found by gradient descent: w: [0.08 0.54 0.03], b: 0.0106
	```
- the model after fitting/training is: 0.08x + 0.54x^2 + 0.03x^3 + 0.0106
- GD has emphasized the data that is the best fit to the x^2 data by increasing the w_1 term relative to the others
- if you were to run for a very long time, it would continue to reduce the impact of the other terms
- GD is picking the ‘correct’ features for us by emphasizing its associated parameter
- less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or very close to zero, the associated feature is not useful in fitting the model to the data
- the weight associated with the x^2 feature is much larger than the weights for x or x^3 as it is the most useful in fitting the data

An Alternate View
- polynomial features were chosen based on how well they matched the target data
- another way to think about this is to note that we are still using linear regression once we have created new features
- the best features will be linear relative to the target
- the x^2 feature mapped against the target value y is linear
- linear regression can then easily generate a model using that feature

Scaling features
- if the data set has features with significantly different scales, one should apply feature scaling to speed
- there is x, x^2 and x^3 which will naturally have very different scales
- z-score normalization 
	```python
# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# Peak to Peak range by column in Raw        X:[  19  361 6859]
# Peak to Peak range by column in Normalized X:[3.3  3.18 3.28]
	```
- with more aggressive value of alpha 
	```python
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# Iteration         0, Cost: 9.42147e+03
# Iteration     10000, Cost: 3.90938e-01
# Iteration     20000, Cost: 2.78389e-02
# Iteration     30000, Cost: 1.98242e-03
# Iteration     40000, Cost: 1.41169e-04
# Iteration     50000, Cost: 1.00527e-05
# Iteration     60000, Cost: 7.15855e-07
# Iteration     70000, Cost: 5.09763e-08
# Iteration     80000, Cost: 3.63004e-09
# Iteration     90000, Cost: 2.58497e-10
# w,b found by gradient descent: w: [5.27e-05 1.13e+02 8.43e-05], b: 123.5000
	```
- feature scaling allows this to converge much faster
- the w_1 term, which is the x^2 term, is the most emphasized

Complex Functions
- with feature engineering, even quite complex functions can be modeled
```python
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# Iteration         0, Cost: 2.20188e-01
# Iteration    100000, Cost: 1.70074e-02
# Iteration    200000, Cost: 1.27603e-02
# Iteration    300000, Cost: 9.73032e-03
# Iteration    400000, Cost: 7.56440e-03
# Iteration    500000, Cost: 6.01412e-03
# Iteration    600000, Cost: 4.90251e-03
# Iteration    700000, Cost: 4.10351e-03
# Iteration    800000, Cost: 3.52730e-03
# Iteration    900000, Cost: 3.10989e-03
# w,b found by gradient descent: w: [ -1.34 -10.    24.78   5.96 -12.49 -16.26  -9.51   0.59   8.7   11.94
#    9.27   0.79 -12.82], b: -0.0073
```

### Optional Lab: Linear Regression using Scikit-Learn

- there is an open-source, commercially usable machine learning toolkit caller scikit-learn
- contains implementations of many of the algorithms
- some imports 
```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
```

GD
- scikit-learn has a GD regression model → sklearn.linear_model.SGDRegressor
- model performs best with normalized inputs
- sklearn.preprocessing.StandardScaler will perform z-score normalization
- scale the training data 
	```python
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train) # combines fit() (mean) and transform()
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# Peak to Peak range by column in Raw        X:[2.41e+03 4.00e+00 1.00e+00 9.50e+01]
# Peak to Peak range by column in Normalized X:[5.85 6.14 2.06 3.69]
	```
- create and fit the regression model 
	```python
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
#              eta0=0.01, fit_intercept=True, l1_ratio=0.15,
#              learning_rate='invscaling', loss='squared_loss', max_iter=1000,
#              n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
#              shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
#              warm_start=False)
# number of iterations completed: 123, number of weight updates: 12178.0
	```
- should also take notes on the scikitlearn page
- understanding parameters for the SGDRegressor
	- loss : str, default = ‘squared_error’
		- the loss function to be used
		- possible values are ‘squared_error’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
		- squared_error → the ordinary least squares fit
		- huber -\> modifies ‘squared_error’ to focus less on getting outlies correct by switching from squared to linear loss past a distance of epsilon
		- epsion_insensitive → ignores errors less than epsilon and is linear past that; this is the loss function used in SVR (support vector regressor?)
		- squared_epsilon_insensitive → is the same but becomes squared loss past a tolerance of epsilon
	- will review the rest when i study scikitlearn
- view parameters 
```python
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# model parameters:                   w: [110.13 -21.02 -32.42 -38.06], b:[363.15]
# model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16
```
- make predictions 
- use both the predict routine and compute using w and b
	```python
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# prediction using np.dot() and sgdr.predict match: True
# Prediction on training set:
# [295.13 485.96 389.59 492.12]
# Target values 
# [300.  509.8 394.  540. ]Plot Results 
	```
- plot predictions
	```python
# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
	```

### Programming Assignment: Week 2 practice lab: Linear regression

Packages
```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
%matplotlib inline
```

Problem Statement
- Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet
	- you would like to expand your business to cities that may give your restaurant higher profits
	- the chain already has restaurants in various cities and you have data for profits and populations from the cities
	- you also have data on cities that are candidates for a new restaurant
		- for these cities, you have the city population
- Can you use the data to help you identify which cities may potentially give your business higher profits?

Dataset
- The load_data() function shown below loads the data into variables x_train and y_train
	- x_train is the population of a city
	- y_train is the profit of a restaurant in that city
	- a negative value for profit indicates a loss
	- both x_train and y_train are numppy arrays 
	```python
# load the dataset
x_train, y_train = load_data() # from utils file
	```
- Viewing the variables
	- it is useful to get more familiar with our dataset
		- a good place to start is to just print out each variable and see what it contains
		```python
# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 

# Type of x_train: <class 'numpy.ndarray'>
# First five elements of x_train are:
#  [6.1101 5.5277 8.5186 7.0032 5.8598]
		```
	- x_train is a numpy array that contains decimal values that are all greater than zero
		- these values represent the city population times 10,000
		- for example, 6.1101 means that the population for that city is 61,101
	- now printing y_train 
		```python
# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  

# Type of y_train: <class 'numpy.ndarray'>
# First five elements of y_train are:
#  [17.592   9.1302 13.662  11.854   6.8233]
		```
	- y_train is a numpy array that has decimal values, some negative, some positive
	- these represent your restaurant’s average monthly profits in each city, in units of \$10,000
		- for example, 17.592 represents %175,920 in average monthly profits for that city
		- -2.6807 represents -\$26,807 in average monthly loss for that city

- Check the dimensions of your variables
	- another useful way to get familiar with your data is to view its dimensions
	- print the shape of x_train and y_train and see how any training examples you have in your dataset 
		```python
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))

# The shape of x_train is: (97,)
# The shape of y_train is:  (97,)
# Number of training examples (m): 97
		```
	- the city population array has 97 data points, and the monthly average profits also has 97 data points
	- these are numpy 1D arrays

- Visualize your data
	- it is often useful to understand the data by visualizing it
	- for this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population) 
		```python
# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
		```
		- your goal is to build a linear regression model to fit this data
	- with this model, you can then input a new city’s population, and have the model estimate your restaurant’s potential monthly profits for that city

Refresher on linear regression
- fit the linear regression parameters (w, b) to your dataset
- the model function for linear regression, which is a function that maps from x (city population) to y (your restaurant’s monthly profit for that city) is represented as f_w,b = wx + b
- to train a linear regression model, you wan to find the best (w,b) parameters that fit your dataset
	- to compare how one choice of (w,b) is better or worse than another choice, you can evaluate it with a cost function J(w,b)
	- J is a function of (w,b), that is, the value of the cost J(w,b)  depends on the value of (w,b)
	- the choice of (w,b) that fits your data the best is the one that has the smallest cost J(w,b)
- to find the values of (w,b) that gets the smallest possible cost J(w,b), you can use a method called gradient descent
	- with each step of gradient descent, your parameters (w,b) come closer to th eoptimal values that will achieve the lowest cost J(w,b)
- the trained linear regression model can then take the input feature x (city population) and output a prediction f_w,b(x) (predicted monthly profit for a restaurant in that city)

Compute Cost
- GD involves repeated steps to adjust the value of your parameter () to gradually tet a smaller and smaller cost J(w,b)w,b
	- at each step of gradient descent, it will be helpful for you to monitor your progress by computing the cost J(w,b) as (w,b) gets updates.
	- you will implement a function to calculate J(w,b) so that you can check the progress of you GD implementation
- Cost function 
	- the cost function for linear regression J(w,b) is defined as
		- you can think of f_w,b(x^i) as the model’s prediction of your restaurant’s profit, as opposed to y^i, which is the actual profit that is recorded in the data
- Model prediction
	- for linear regression with one variable, the predicition of the model f_w,b for an example x^i is represented as f_w,b(x^i) = wx(i) + b
	- this is the equation for a line, with an intercept b and a slope w
- Implementation
	- complete the compute_cost() function below to compute the cost J(w,b)

Exercise 1
- complete the compute_cost below to:
	- iterate over the training examples, and for each example, compute:
		- the prediction of the model for that example → f_w,b(x^i) = wx^i + b
		- the cost for that example → cost^i = (f_w,b - y^i)^2
	- return the total cost over all examples
		- here, m is the number of training examples
			```python
# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    
    for i in range(m)
	    f_wb = w*x[i] + b
	    total_cost = total_cost + (f_wb - y[i])**2
    
	  total_cost = total_cost / (2*m)
	   
    ### END CODE HERE ### 

    return total_cost
	```
- Check by running the following test code 
	```python
# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
from public_tests import *
compute_cost_test(compute_cost)

# <class 'numpy.float64'>
# Cost at initial w: 75.203
# All tests passed!
	```
- Expected Output → cost at initial w: 75.203

Gradient descent
- as described in the lecture videos, the gradient descent algorithm is
- where, parameter w, b are both updated simultaneously and where
	- m is th enumber of training examples in the dataset
	- f_w,b(x^i) is the model’s prediction, while y^i, is the target value
- implement a function called compute_gradient which calculates dJ(w)/dw, dJ(w)/db

Exercise 2
- complete the compute_gradient function
	- iterate over the training example, and for each example, compute:
		- the prediction of the mmodel for that example → f_w,b (x^i) = wx^i + b
		- the gradient for the parameters w, b from that example
			- return the total gradient update from all the examples
		- here, m is the number of training examples
				```python
# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]
        
    dj_dw /= m
    dj_db /= m
    
    ### END CODE HERE ### 
        
    return dj_dw, dj_db
		```
	- check your implementation of the compute_gradient function with two different initializations of the parameters w, b 
		```python
# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

compute_gradient_test(compute_gradient)

# Gradient at initial w, b (zeros): -65.32884974555672 -5.83913505154639
# Using X with shape (4, 1)
# All tests passed!
		```
		```python
# Compute and display cost and gradient with non-zero w
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)

# Gradient at test w, b: -47.41610118114435 -4.007175051546391
		```
- Learning parameters using batch gradient descent
	- find the optimal parameters of a linear regression model by using batch gradient descent
	- recall batch refers to running all the examples in one iteration
		- a good way to verify that GD is working correctly is to look at the value of J(w, b) and check that it is decreasing with each step
		- assuming you have implemented the gradient and computed the cost correctly and you have an appropriate value for the learning rate alpha, J(w, b) should never increase and should converge to a steady value by the end of the alg 
		```python
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing
		```
		```python
# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

# Iteration    0: Cost     6.74   
# Iteration  150: Cost     5.31   
# Iteration  300: Cost     4.96   
# Iteration  450: Cost     4.76   
# Iteration  600: Cost     4.64   
# Iteration  750: Cost     4.57   
# Iteration  900: Cost     4.53   
# Iteration 1050: Cost     4.51   
# Iteration 1200: Cost     4.50   
# Iteration 1350: Cost     4.49   
# w,b found by gradient descent: 1.166362350335582 -3.63029143940436
		```
	- we will now use the final parameters from gradient descent to plot th elinear fit
	- recall that we can get the prediction for a single example f(x^i) = wx^i + b
	- to calculate the prediction for each example 
		```python
m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b
		```
		```python
# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
		```
		- your final values of w, b can also be used to make predictions on profits
	- predict what the profit would be in areas of 35,000 and 70,000 people
		- the model takes in population of a city in 10,000s as input
		- therefore, 35,000 people can be translated into an input to the model as np.array([3.5])
		- similarly, 70,000 people can be translated into an input to the model as np.array([7.])
		```python
predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

# For population = 35,000, we predict a profit of $4519.77
# For population = 70,000, we predict a profit of $45342.45
		```