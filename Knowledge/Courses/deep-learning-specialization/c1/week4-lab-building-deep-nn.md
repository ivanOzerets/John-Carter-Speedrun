Packages

```python
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from public_tests import *

import copy
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
```

Outline

- implement several helper functions
    - initialize the parameters for a two-layer network and for an L-layer NN
    - implement the forward prop module
        - complete the LINEAR part of the layer’s forward prop step (resulting in Z[l])
        - the ACTIVATION function is provided
        - combine the previous two steps into a new [LINEAR→ACTIVATION] forward function
        - stack the [LINEAR→RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR→SIGMOID] at the end (for the final layer L)
        - this gives a new L_model_forward function
    - compute the loss
    - implement the backward prop module
        - complete the LINEAR part of a layers’ backward prop setp
        - the gradient of the ACTIVATION function is provided
        - combine the previous two steps into a new [LINEAR→ACTIVATION] backward function
        - stack[LINEAR→RELU] backward L-1 times and add [LINEAR→SIGMOID] backward in a new L_model_backward function
    - update parameters

    - for every forward function, there is a corresponding backward function
    - this is why at every step of the forward module, store some values in a cache
    - these cached values are useful for computing gradients
    - in the backprop module, use the cache to calculate the gradients

Initialization

- write two helper functions to initialize the parameters of the model
- the first function will be used to initialize parameters for a tow layer model
- the second one generalizes this initiation process to L layers
- 2-layer NN
    - Exercise 1 - initialize_parameters
        - create and initialize the parameters of the 2-layer network
        - the model’s structure is: LINEAR→RELU→LINEAR→SIGMOID
        - use the random initialization for the weight matrices: np.random.randn(d0, d1, .., dn) * 0.01 with the correct shape
        - use zero initialization for the biases: np.zeros(shape)

        ```python
        # GRADED FUNCTION: initialize_parameters

        def initialize_parameters(n_x, n_h, n_y):
            """
            Argument:
            n_x -- size of the input layer
            n_h -- size of the hidden layer
            n_y -- size of the output layer

            Returns:
            parameters -- python dictionary containing your parameters:
                            W1 -- weight matrix of shape (n_h, n_x)
                            b1 -- bias vector of shape (n_h, 1)
                            W2 -- weight matrix of shape (n_y, n_h)
                            b2 -- bias vector of shape (n_y, 1)
            """

            np.random.seed(1)

            #(≈ 4 lines of code)
            # W1 = ...
            # b1 = ...
            # W2 = ...
            # b2 = ...
            # YOUR CODE STARTS HERE
            W1 = np.random.randn(n_h, n_x) * 0.01
            b1 = np.zeros((n_h, 1))
            W2 = np.random.randn(n_y, n_h) * 0.01
            b2 = np.zeros((n_y, 1))
            # YOUR CODE ENDS HERE

            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}

            return parameters    
        ```

- L-layer NN
    - the initialization for a deeper L-layer NN is more complicated because there are many more weight matrices and bias vectors
    - when completing the initialize_parameters_deep function, make sure that the dimensions match between each layer
    - n[l] is the number of units in layer l
    - if the size of the input X is (12288, 209), then:

    - Exercise 2 - initialize_parameters_deep
        - implement initialization for an L-layer NN
        - the model’s structure is [LINEAR→RELU] x (L-1) → LINEAR → SIGMOID, it has L-1 layers using a ReLU activation function followed by an output layer with a sigmoid activation function
        - use random initialization for the weight matrices, use np.random.randn(0, d1, .., dn) * 0.01
        - use zeros initialization for the biases, np.zeros(shape)
        - store n[l], the nmber of units in different layers, in a variable layer_dims
        - the implementation for L=1 (one layer NN)

            ```python
            if L == 1:
            			parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
                  parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
            ```

        ```python
        # GRADED FUNCTION: initialize_parameters_deep

        def initialize_parameters_deep(layer_dims):
            """
            Arguments:
            layer_dims -- python array (list) containing the dimensions of each layer in our network

            Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                            bl -- bias vector of shape (layer_dims[l], 1)
            """

            np.random.seed(3)
            parameters = {}
            L = len(layer_dims) # number of layers in the network

            for l in range(1, L):
                #(≈ 2 lines of code)
                # parameters['W' + str(l)] = ...
                # parameters['b' + str(l)] = ...
                # YOUR CODE STARTS HERE
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
                parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
                # YOUR CODE ENDS HERE

                assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
                assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

            return parameters
        ```

Forward Propagation Module

- Linear Forward
    - write the forward prop module
    - start by implementing some basic functions
        - LINEAR
        - LINEAR→ACTIVATION where ACTIVATION will be either ReLU or Sigmoid
        - [LINEAR→ReLU] x (L-1) → LINEAR → SIGMOID
    - the linear forward module computes the following equations

    - where A[0]= X
    - Exercise 3 = linear_forward
        - build the linear part of forward prop
        - the mathematical representation of this unit is Z[l] = W[l]A[l-1] + b[l]

        ```python
        # GRADED FUNCTION: linear_forward

        def linear_forward(A, W, b):
            """
            Implement the linear part of a layer's forward propagation.

            Arguments:
            A -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)

            Returns:
            Z -- the input of the activation function, also called pre-activation parameter 
            cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
            """

            #(≈ 1 line of code)
            # Z = ...
            # YOUR CODE STARTS HERE
            Z = np.dot(W, A) + b    
            # YOUR CODE ENDS HERE
            cache = (A, W, b)

            return Z, cache
        ```

- Linear-Activation Forward
    - sigmoid
        - return two items, the activation value a and a cache that contains Z
        - call using A, activation_cache = sigmoid(Z)

    - ReLU
        - return two items, the activation value A and a cache with Z
        - call the same way

    - to group linear and activation, implement a function that does the LINEAR forward step, followed by an ACTIVATION step
    - Exercise 4 - linear_activation_forward
        - implement the forward prop of the LINEAR→ACTIVATION layer
        - mathematical relation is A[l] = g(Z[l]) = g(W[l]A[l-1] + b[l]) where the activation g can be sigmoid or relu

        ```python
        # GRADED FUNCTION: linear_activation_forward

        def linear_activation_forward(A_prev, W, b, activation):
            """
            Implement the forward propagation for the LINEAR->ACTIVATION layer

            Arguments:
            A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

            Returns:
            A -- the output of the activation function, also called the post-activation value 
            cache -- a python tuple containing "linear_cache" and "activation_cache";
                     stored for computing the backward pass efficiently
            """

            if activation == "sigmoid":
                #(≈ 2 lines of code)
                # Z, linear_cache = ...
                # A, activation_cache = ...
                # YOUR CODE STARTS HERE
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = sigmoid(Z)
                # YOUR CODE ENDS HERE

            elif activation == "relu":
                #(≈ 2 lines of code)
                # Z, linear_cache = ...
                # A, activation_cache = ...
                # YOUR CODE STARTS HERE
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = relu(Z)
                # YOUR CODE ENDS HERE
            cache = (linear_cache, activation_cache)

            return A, cache
        ```

    - in deep learning, the [LINEAR→ACTIVATION] computation is counted as a single layer in the NN, not two layers
- L-Layer Model
    - for even more convenience when implementing the L-layer NN, write a function that replicates the previous one (linear_activation_forward with RELU) L - 1 times, then follows that with one linear_activation_forward with SIGMOID

    - Exercise 5 - L_model_forward
        - implement the forward prop of the above model
        - the variable AL will denote A[L] = sigmoid(Z) = sigmoid(W[L]A[L-1] + b[L]), sometimes called Yhat
        - keep track of the caches int he caches list
        - to add a new value c to a list, use list.append(c)

        ```python
        # GRADED FUNCTION: L_model_forward

        def L_model_forward(X, parameters):
            """
            Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

            Arguments:
            X -- data, numpy array of shape (input size, number of examples)
            parameters -- output of initialize_parameters_deep()

            Returns:
            AL -- activation value from the output (last) layer
            caches -- list of caches containing:
                        every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
            """

            caches = []
            A = X
            L = len(parameters) // 2                  # number of layers in the neural network

            # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
            # The for loop starts at 1 because layer 0 is the input
            for l in range(1, L):
                A_prev = A 
                #(≈ 2 lines of code)
                # A, cache = ...
                # caches ...
                # YOUR CODE STARTS HERE
                A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
                caches.append(cache)
                # YOUR CODE ENDS HERE

            # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
            #(≈ 2 lines of code)
            # AL, cache = ...
            # caches ...
            # YOUR CODE STARTS HERE
            AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
            caches.append(cache)
            # YOUR CODE ENDS HERE

            return AL, caches
        ```

    - takes input X and outputs a row vector A[L] containing the predictions
    - also records all intermediate values in ‘caches’
    - using A[L], compute the cost of the predictions

Cost Function

- implement forward and backward prop
- compute the cost, in order to check whether the model is learning
- Exercise 6 - compute_cost
    - compute the cross-entropy cost J

    ```python
    # GRADED FUNCTION: compute_cost

    def compute_cost(AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        # (≈ 1 lines of code)
        # cost = ...
        # YOUR CODE STARTS HERE
        cost = - 1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)), axis = 1, keepdims = True)    
        # YOUR CODE ENDS HERE

        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost
    ```

Backward Prop Module

- back prop is used to calculate the gradient of the loss function wrt the parameters

- LINEAR backward
- LINEAR → ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
- [LINEAR→RELU] x (L-1) → LINEAR → SIGMOID backward
- b is a matrix with 1 column and n rows
- np.sum performs a sum over the elements of a ndarray
- axis=1 or axis=0 specify if the sum is carried out by rows or by columns respectively
- keepdims specifies if the original dimensions of the matrix must be kept

    ```python
    A = np.array([[1, 2], [3, 4]])

    print('axis=1 and keepdims=True')
    print(np.sum(A, axis=1, keepdims=True))
    print('axis=1 and keepdims=False')
    print(np.sum(A, axis=1, keepdims=False))
    print('axis=0 and keepdims=True')
    print(np.sum(A, axis=0, keepdims=True))
    print('axis=0 and keepdims=False')
    print(np.sum(A, axis=0, keepdims=False))

    '''
    axis=1 and keepdims=True
    [[3]
     [7]]
    axis=1 and keepdims=False
    [3 7]
    axis=0 and keepdims=True
    [[4 6]]
    axis=0 and keepdims=False
    [4 6]
    '''
    ```

- Linear Backward
    - for layer l, the linear part is: Z[l] = W[l]A[l-1] + b[k] + activation
    - suppose you have dZ[l] = dL/dZ[l] and you want (dW[l], db[l], dA[l-1])

    - the three outputs are computed using the input dZ[l]

    - Exercise 7 - linear_backward
        - implement linear_backward()

        ```python
        # GRADED FUNCTION: linear_backward

        def linear_backward(dZ, cache):
            """
            Implement the linear portion of backward propagation for a single layer (layer l)

            Arguments:
            dZ -- Gradient of the cost with respect to the linear output (of current layer l)
            cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

            Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
            A_prev, W, b = cache
            m = A_prev.shape[1]

            ### START CODE HERE ### (≈ 3 lines of code)
            # dW = ...
            # db = ... sum by the rows of dZ with keepdims=True
            # dA_prev = ...
            # YOUR CODE STARTS HERE
            dW = 1/m * np.dot(dZ, A_prev.T)
            db = 1/m * np.sum(dZ, axis = 1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)
            # YOUR CODE ENDS HERE

            return dA_prev, dW, db
        ```

- Linear-Activation Backward
    - create a function that merges the two helper functions: linear_backward and the backward step for the activation linear_activation_backward
    - sigmoid_backward: implements the backward propagation for SIGMOID unit
        - dZ = sigmoid_backward(dA, activation_cache)
    - relu_backward: implements the backward propagation for RELU unit
        - dZ = relu_backward(dA, activation_cache)
    - if g() is the activation function, sigmoid_backward and relu_backward compute
        - dZ[l] = dA[l] * g’(Z[l])
    - Exercise 8 - linear_activation_backward
        - implement the backprop for the LINEAR→ACTIVATION layer

        ```python
        # GRADED FUNCTION: linear_activation_backward

        def linear_activation_backward(dA, cache, activation):
            """
            Implement the backward propagation for the LINEAR->ACTIVATION layer.

            Arguments:
            dA -- post-activation gradient for current layer l 
            cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

            Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
            linear_cache, activation_cache = cache

            if activation == "relu":
                #(≈ 2 lines of code)
                # dZ =  ...
                # dA_prev, dW, db =  ...
                # YOUR CODE STARTS HERE
                dZ = relu_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)
                # YOUR CODE ENDS HERE

            elif activation == "sigmoid":
                #(≈ 2 lines of code)
                # dZ =  ...
                # dA_prev, dW, db =  ...
                # YOUR CODE STARTS HERE
                dZ = sigmoid_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)        
                # YOUR CODE ENDS HERE

            return dA_prev, dW, db
        ```

- L-Model Backward
    - implement the backward function for the whole network
    - the L_model_forward function, at each iteration, stored a cache which contains (X, Q, b, and z)
    - in the L_model_forward function, iterate through all the hidden layers backward, starting from layer L
    - on each step, use the cached values for layer l to backpropagate through layer l

- initializing backpropagation
    - to backprop through the network, know that the output is A[L] = sigmoid(Z[L])
    - the code thus needs to compute dAL = dL/dA[L]
    - use the formula

    - use this post-activation gradient dAL to keep going backward
    - feed in dAL into the LINEAR→SIGMOID backward function
    - use a for loop to iterate through all the other layers using the LINEAR→RELU backward function
    - should store each dA, dW, and db in the grads dictionary

    - Exercise 9 - L_model_backward
        - implement backprop for the [LINEAR→RELU] x (L-1) → LINEAR → SIGMOID model

        ```python
        # GRADED FUNCTION: L_model_backward

        def L_model_backward(AL, Y, caches):
            """
            Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

            Arguments:
            AL -- probability vector, output of the forward propagation (L_model_forward())
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
            caches -- list of caches containing:
                        every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                        the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

            Returns:
            grads -- A dictionary with the gradients
                     grads["dA" + str(l)] = ... 
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ... 
            """
            grads = {}
            L = len(caches) # the number of layers
            m = AL.shape[1]
            Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

            # Initializing the backpropagation
            #(1 line of code)
            # dAL = ...
            # YOUR CODE STARTS HERE
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            # YOUR CODE ENDS HERE

            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
            #(approx. 5 lines)
            # current_cache = ...
            # dA_prev_temp, dW_temp, db_temp = ...
            # grads["dA" + str(L-1)] = ...
            # grads["dW" + str(L)] = ...
            # grads["db" + str(L)] = ...
            # YOUR CODE STARTS HERE
            current_cache = caches[L-1]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
            grads["dA" + str(L-1)] = dA_prev_temp
            grads["dW" + str(L)] = dW_temp
            grads["db" + str(L)] = db_temp
            # YOUR CODE ENDS HERE

            # Loop from l=L-2 to l=0
            for l in reversed(range(L-1)):
                # lth layer: (RELU -> LINEAR) gradients.
                # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
                #(approx. 5 lines)
                # current_cache = ...
                # dA_prev_temp, dW_temp, db_temp = ...
                # grads["dA" + str(l)] = ...
                # grads["dW" + str(l + 1)] = ...
                # grads["db" + str(l + 1)] = ...
                # YOUR CODE STARTS HERE
                current_cache = caches[l]
                dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
                grads["dA" + str(l)] = dA_prev_temp
                grads["dW" + str(l+1)] = dW_temp
                grads["db" + str(l+1)] = db_temp        
                # YOUR CODE ENDS HERE

            return grads
        ```

- Update Parameters
    - update the parameters of the model, using GD

    - after computing, store in parameters dictionary
    - Exercise 10 - update_parameters
        - implement update_parameters() to update the parameters using GD
        - on every W[l] and b[l] for l = 1, …, L

        ```python
        # GRADED FUNCTION: update_parameters

        def update_parameters(params, grads, learning_rate):
            """
            Update parameters using gradient descent

            Arguments:
            params -- python dictionary containing your parameters 
            grads -- python dictionary containing your gradients, output of L_model_backward

            Returns:
            parameters -- python dictionary containing your updated parameters 
                          parameters["W" + str(l)] = ... 
                          parameters["b" + str(l)] = ...
            """
            parameters = copy.deepcopy(params)
            L = len(parameters) // 2 # number of layers in the neural network

            # Update rule for each parameter. Use a for loop.
            #(≈ 2 lines of code)
            for l in range(L):
                # parameters["W" + str(l+1)] = ...
                # parameters["b" + str(l+1)] = ...
                # YOUR CODE STARTS HERE
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
                # YOUR CODE ENDS HERE
            return parameters
        ```