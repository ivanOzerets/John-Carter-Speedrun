Building basic functions with numpy

- learn several key numpy functions such as np.exp, np.log, and np.reshape
- sigmoid function, np.exp()
    - first try using math.exp() and the np.exp() to see the difference
    - Exercise 2 - basic_sigmoid
        - build a function that returns the sigmoid of a real number x
        - use math.exp(x) for the exponential function
        - sigmoid(x) = 1/(1+e^-x) is sometimes also known as the logistic function
        - it is a non-linear function used in ML and Deep Learning

        - to refer to a function berlonging to a specific package, call it using package_name.function()

            ```python
            import math
            from public_tests import *

            # GRADED FUNCTION: basic_sigmoid

            def basic_sigmoid(x):
                """
                Compute sigmoid of x.

                Arguments:
                x -- A scalar

                Return:
                s -- sigmoid(x)
                """
                # (≈ 1 line of code)
                # s = 
                # YOUR CODE STARTS HERE
                s = 1 / (1 + math.exp(-x))
                # YOUR CODE ENDS HERE

                return s
            ```

        - the math library is rarely used in dep learning because the inputs of the functions are real numbers
        - deep learning mostly uses matrices and vectors

            ```python
            ### One reason why we use "numpy" instead of "math" in Deep Learning ###

            x = [1, 2, 3] # x becomes a python list object
            basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.
            ```

        - if x=(x1,x2,…,xn) is a row vector, then np.exp(x) will apply the exponential function to every elemnt of x
        - the output will thus be np.exp(x) = (e^{x_1}, …, e^{x_n})

            ```python
            import numpy as np

            # example of np.exp
            t_x = np.array([1, 2, 3])
            print(np.exp(t_x)) # result is (exp(1), exp(2), exp(3))

            # [ 2.71828183  7.3890561  20.08553692]
            ```

        - if x is a vector, then a python operation such as s = x+3 or s = 1/x will output s as a vector of the same size as x

            ```python
            # example of vector operation
            t_x = np.array([1, 2, 3])
            print (t_x + 3)

            # [4 5 6]
            ```

    - Exercise 3 - sigmoid
        - implement the sigmoid function using numpy
        - x could now be either a real number, a vector, or a matrix
        - the data structures in numpy used to represent the shapes of vectors, matrices.. are called numpy arrays

            ```python
            # GRADED FUNCTION: sigmoid

            def sigmoid(x):
                """
                Compute the sigmoid of x

                Arguments:
                x -- A scalar or numpy array of any size

                Return:
                s -- sigmoid(x)
                """

                # (≈ 1 line of code)
                # s = 
                # YOUR CODE STARTS HERE
                s = 1 / (1+np.exp(-x))
                # YOUR CODE ENDS HERE

                return s
            ```

- Sigmoid Gradient
    - computing gradients to optimize loss function using back propagation
    - Exercise 4 - sigmoid_derivative
        - implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x

        - code in two steps
            - set s to be the sigmoid of x
            - compute gradient of sigmoid(x) = s(1-s)

        ```python
        # GRADED FUNCTION: sigmoid_derivative

        def sigmoid_derivative(x):
            """
            Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
            You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

            Arguments:
            x -- A scalar or numpy array

            Return:
            ds -- Your computed gradient.
            """

            #(≈ 2 lines of code)
            # s = 
            # ds = 
            # YOUR CODE STARTS HERE
            s = sigmoid(x)
            ds = s*(1-s)
            # YOUR CODE ENDS HERE

            return ds
        ```

- Reshaping arrays
    - two common numpy used in deep learning are np.shape and np.reshape()
        - x.shape is used to get the shape (dimension) of a matrix/vector X
        - X.reshape() is used to reshape X into some other dimension
    - for example, in comp sci, an image is represented by a 3D array of shape (length, height, depth = 3)
    - when you read an image as the input of an alg you convert it to a vector of shape (length x height x 3, 1)
    - in other words, you ‘unroll’, or reshape, the 3D array into a 1D vector

    - Exercise 5 - image2vector
        - implement image2vector() that takes an input of shape (length, height, 3) and returns a vector of shape (length x height x 3, 1)
        - if you would like to reshape an array v of shape (a,b,c) into a vector of shape (a x b, c)
            - v = v.reshape((v.shape[0] x v.shape[1], v.shape[2]))
            - don’t hardcode dimensions of an image as a constant
            - look up the quantities with image.shape[0]
            - use v = v.reshape(-1,1)

        ```python
        # GRADED FUNCTION:image2vector

        def image2vector(image):
            """
            Argument:
            image -- a numpy array of shape (length, height, depth)

            Returns:
            v -- a vector of shape (length*height*depth, 1)
            """

            # (≈ 1 line of code)
            # v =
            # YOUR CODE STARTS HERE
            v = image.reshape(-1,1) # -1 is a wildcard
            # YOUR CODE ENDS HERE

            return v
        ```

- Normalizing rows
    - another common technique in ML and DL is to normalize the data
    - leads to better performance because gradient descent converges faster after normalization
    - by normalization, change x to x/||x|| (dividing each row vector of x by its norm)

    - you can divide matrices of different sizes
    - with keepdims = true, the result will broadcast correctly against the original x, meaning keeps one of the dimensions i think
    - axis=1 means take the norm in a row-wise manner
    - numpy.linalg.norm has another parameter ord to specify the type of normalization to be done
    - Exercise 6 - normalize_rows
        - implement normalizeRows() to normalize the rows of a matrix
        - after applying the function to an input matrix x, each row of x should be a vector of unit length 1

        ```python
        # GRADED FUNCTION: normalize_rows

        def normalize_rows(x):
            """
            Implement a function that normalizes each row of the matrix x (to have unit length).

            Argument:
            x -- A numpy matrix of shape (n, m)

            Returns:
            x -- The normalized (by row) numpy matrix. You are allowed to modify x.
            """

            #(≈ 2 lines of code)
            # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
            # x_norm =
            # Divide x by its norm.
            # x =
            # YOUR CODE STARTS HERE
            x_norm = np.linalg.norm(x,axis=1,keepdims=True)
            x=x / x_norm
            # YOUR CODE ENDS HERE

            return x
        ```

        - in normalize_rows(), trying to print the shapes of x_norm because of broadcasting
    - Exercise 7 - softmax
        - implement a softmax function using numpy
        - think of softmax as a normalizing function used when the alg needs to classify two or more classes

        - m will b e later used to represent the number of training examples, and each training example is in its own column of the matrix
        - each feature will be in its own row
        - softmax should be performed for all features of each training example, so softmax would be performed on the columns

        ```python
        # GRADED FUNCTION: softmax

        def softmax(x):
            """Calculates the softmax for each row of the input x.

            Your code should work for a row vector and also for matrices of shape (m,n).

            Argument:
            x -- A numpy matrix of shape (m,n)

            Returns:
            s -- A numpy matrix equal to the softmax of x, of shape (m,n)
            """

            #(≈ 3 lines of code)
            # Apply exp() element-wise to x. Use np.exp(...).
            # x_exp = ...

            # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
            # x_sum = ...

            # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
            # s = ...

            # YOUR CODE STARTS HERE
            x_exp = np.exp(x)
            x_sum = np.sum(x_exp, axis=1, keepdims=True)
            s = x_exp/x_sum
            # YOUR CODE ENDS HERE

            return s
        ```

        - printing the shapes of x_exp, x_sum and s will shows broadcasting in action
        - np.exp(x) works for any np.array x and applies the exponential function to every coordinate

Vectorization

- in deep learning, there are very large datasets
- a non-computationally-optimal function can become a huge bottleneck in the alg and can result in the model that takes ages to run
- to make sure that the code is computationally efficient, use vectorization
- what is the difference between the following implementations of the dot/outer/elementwise product

    ```python
    import time

    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

    ### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
    tic = time.process_time()
    dot = 0

    for i in range(len(x1)):
        dot += x1[i] * x2[i]
    toc = time.process_time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
    tic = time.process_time()
    outer = np.zeros((len(x1), len(x2))) # we create a len(x1)*len(x2) matrix with only zeros

    for i in range(len(x1)):
        for j in range(len(x2)):
            outer[i,j] = x1[i] * x2[j]
    toc = time.process_time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### CLASSIC ELEMENTWISE IMPLEMENTATION ###
    tic = time.process_time()
    mul = np.zeros(len(x1))

    for i in range(len(x1)):
        mul[i] = x1[i] * x2[i]
    toc = time.process_time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
    W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
    tic = time.process_time()
    gdot = np.zeros(W.shape[0])

    for i in range(W.shape[0]):
        for j in range(len(x1)):
            gdot[i] += W[i,j] * x1[j]
    toc = time.process_time()
    print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    '''
    dot = 278
     ----- Computation time = 0.0913090000000949ms
    outer = [[81. 18. 18. 81.  0. 81. 18. 45.  0.  0. 81. 18. 45.  0.  0.]
     [18.  4.  4. 18.  0. 18.  4. 10.  0.  0. 18.  4. 10.  0.  0.]
     [45. 10. 10. 45.  0. 45. 10. 25.  0.  0. 45. 10. 25.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [63. 14. 14. 63.  0. 63. 14. 35.  0.  0. 63. 14. 35.  0.  0.]
     [45. 10. 10. 45.  0. 45. 10. 25.  0.  0. 45. 10. 25.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [81. 18. 18. 81.  0. 81. 18. 45.  0.  0. 81. 18. 45.  0.  0.]
     [18.  4.  4. 18.  0. 18.  4. 10.  0.  0. 18.  4. 10.  0.  0.]
     [45. 10. 10. 45.  0. 45. 10. 25.  0.  0. 45. 10. 25.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
     ----- Computation time = 0.27300900000004624ms
    elementwise multiplication = [81.  4. 10.  0.  0. 63. 10.  0.  0.  0. 81.  4. 25.  0.  0.]
     ----- Computation time = 0.12061499999993508ms
    gdot = [23.07978058 13.60794274 20.06438432]
     ----- Computation time = 0.36652500000000643ms
    '''
    ```

    ```python
    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

    ### VECTORIZED DOT PRODUCT OF VECTORS ###
    tic = time.process_time()
    dot = np.dot(x1,x2)
    toc = time.process_time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### VECTORIZED OUTER PRODUCT ###
    tic = time.process_time()
    outer = np.outer(x1,x2)
    toc = time.process_time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### VECTORIZED ELEMENTWISE MULTIPLICATION ###
    tic = time.process_time()
    mul = np.multiply(x1,x2)
    toc = time.process_time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

    ### VECTORIZED GENERAL DOT PRODUCT ###
    tic = time.process_time()
    dot = np.dot(W,x1)
    toc = time.process_time()
    print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    '''
    dot = 278
     ----- Computation time = 0.09018400000004867ms
    outer = [[81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]
     [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [63 14 14 63  0 63 14 35  0  0 63 14 35  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]
     [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
     ----- Computation time = 0.16296700000006936ms
    elementwise multiplication = [81  4 10  0  0 63 10  0  0  0 81  4 25  0  0]
     ----- Computation time = 0.07940799999994752ms
    gdot = [23.07978058 13.60794274 20.06438432]
     ----- Computation time = 0.08142700000002279ms

    '''
    ```

- the vectorized implementation is much cleaner and more efficient
- np.dot() performs a matrix-matrix or matrix-vector multiplication
- this is different from np.multiply() and the * operator, which performs an element-wise multiplication
- Implement the L1 and L2 loss functions
    - Exercise 8 - L1
        - implement the numpy vectorized version of the L1 loss
        - find the function abs(x) is useful
        - the loss is used to evaluate the performance of the model, the bigger th eloss, the more different the predictions (y_hat) are from the true values (y)
        - in deep learning, algs like GD are optimized to train the model and minimize the cost

        ```python
        # GRADED FUNCTION: L1

        def L1(yhat, y):
            """
            Arguments:
            yhat -- vector of size m (predicted labels)
            y -- vector of size m (true labels)

            Returns:
            loss -- the value of the L1 loss function defined above
            """

            #(≈ 1 line of code)
            # loss = 
            # YOUR CODE STARTS HERE
            loss = np.sum(np.abs(y-yhat))    
            # YOUR CODE ENDS HERE

            return loss
        ```

    - Exercise 9 - L2
        - implement the numpy vectorized version of the L2 loss
        - there are several ways of implementing the L2 loss
        - the np.dot() function will be useful
        - if x = [x1, …, xn], then np.dot(x,x) = sum^n_(j=1)(x^2_j)

        ```python
        # GRADED FUNCTION: L2

        def L2(yhat, y):
            """
            Arguments:
            yhat -- vector of size m (predicted labels)
            y -- vector of size m (true labels)

            Returns:
            loss -- the value of the L2 loss function defined above
            """

            #(≈ 1 line of code)
            # loss = ...
            # YOUR CODE STARTS HERE
            loss = np.sum(np.dot(y-yhat, y-yhat))
            # YOUR CODE ENDS HERE

            return loss
        ```