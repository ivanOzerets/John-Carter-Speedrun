Optimization Methods

- stochastic GD
- momentum
- RMSProp
- Adam

Packages

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils_v1a import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils_v1a import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from copy import deepcopy
from testCases import *
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2
```

Gradient Descent

- a simple optimization method in machine learning is GD
- when taking gradient steps w.r.t. all m example on each step, it is also called Batch Gradient Descent
- Exercise 1 - update_parameters_with_gd
    - implement the gradient descent update rule

    - where L is the number of layers and alpha is the learning rate
    - all parameters should be stored in the parameters dictionary
    - the iterator l starts at 1 in the for loop as the first parameters are W[1] and b[1]

    ```python
    # GRADED FUNCTION: update_parameters_with_gd

    def update_parameters_with_gd(parameters, grads, learning_rate):
        """
        Update parameters using one step of gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters to be updated:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients to update each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.

        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        L = len(parameters) // 2 # number of layers in the neural networks

        # Update rule for each parameter
        for l in range(1, L + 1):
            # (approx. 2 lines)
            # parameters["W" + str(l)] =  
            # parameters["b" + str(l)] = 
            # YOUR CODE STARTS HERE
            parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
            # YOUR CODE ENDS HERE
        return parameters
    ```

    - a variant of this is stochastic gradient descent (SGD)
    - equivalent to mini-batch gradient descent, where each mini-batch has just 1 example
    - the update rule does not change
    - what changes is computing gradients on just one training example at a time, rather than on the whole training set

    ```python
    # Batch Gradient Descent

    X = data_input
    Y = labels
    m = X.shape[1]  # Number of training examples
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        # Forward propagation
        a, caches = forward_propagation(X, parameters)
        # Compute cost
        cost_total = compute_cost(a, Y)  # Cost for m training examples
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters
        parameters = update_parameters(parameters, grads)
        # Compute average cost
        cost_avg = cost_total / m

    ```

    ```python
    # SGD

    X = data_input
    Y = labels
    m = X.shape[1]  # Number of training examples
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        cost_total = 0
        for j in range(0, m):
            # Forward propagation
            a, caches = forward_propagation(X[:,j], parameters)
            # Compute cost
            cost_total += compute_cost(a, Y[:,j])  # Cost for one training example
            # Backward propagation
            grads = backward_propagation(a, caches, parameters)
            # Update parameters
            parameters = update_parameters(parameters, grads)
        # Compute average cost
        cost_avg = cost_total / m
    ```

    - in SGD, only 1 training example is used before updating the gradients
    - when the training set is large, SGD can be faster, but the parameters will oscillate toward the minimum rather than converge smoothly

    - SGD requires three for loops
        - over the number of iterations
        - over the m training examples
        - over the layers
    - in practice, faster results come from training on an amount of examples somewhere between 1 and m
    - mini-batch descent uses an intermediate number of examples for each step
    - with mini-batch GD, loop over the mini-batches instead of looping over individual training examples

- Mini-Batch GD
    - build some mini-batches from the training set (X, Y)
        - shuffle
            - create a shuffled version of the training set (X, Y)
            - each column of X and Y represents a training example
            - note that the random shuffling is done synchronously between X and Y
            - such that after the shuffling the ith column of X is the example corresponding to the ith label in Y
            - the shuffling step ensures that examples will be split randomly into different mini-batches
        - partition
            - partition the shuffled (X, Y) into min-batches of size mini_batch_size
            - note that the number of training examples is not always divisible by mini_batch_size
            - the last mini batch might be smaller, but it’s okay

    - Exercise 2 - random_mini_batches
        - implement random_mini_batches
        - the following code selects the indexes for the first and second mini-batches

            ```python
            first_mini_batch_X = shuffled_X[:, 0 : mini_batch_size]
            second_mini_batch_X = shuffled_X[:, mini_batch_size : 2 * mini_batch_size]
            ...
            ```

        - note that the last mini-batch might end up smaller than mini_batch_size=64
        - let floor(s) represent s rounded down to the nearest integer (math.floor(s))
        - if the total number of examples is not a multiple of mini_batch_size=64, then there will be floor(m/mini_batch_size) mini-batches with a full 64 examples and the number of examples in the final mini-batch will be (m - mini_batch_size x floor(m/mini_batch_size)

        - think of a way in which to use the loop variable k to help increment i and j in multiples of mini_batch_size
        - if you want to increment in multiples of 3

        ```python
        # GRADED FUNCTION: random_mini_batches

        def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
            """
            Creates a list of random minibatches from (X, Y)

            Arguments:
            X -- input data, of shape (input size, number of examples)
            Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
            mini_batch_size -- size of the mini-batches, integer

            Returns:
            mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
            """

            np.random.seed(seed)            # To make your "random" minibatches the same as ours
            m = X.shape[1]                  # number of training examples
            mini_batches = []

            # Step 1: Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((1, m))

            inc = mini_batch_size

            # Step 2 - Partition (shuffled_X, shuffled_Y).
            # Cases with a complete mini batch size only i.e each of 64 examples.
            num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                # (approx. 2 lines)
                # mini_batch_X =  
                # mini_batch_Y =
                # YOUR CODE STARTS HERE
                mini_batch_X = shuffled_X[:,k*inc:(k+1)*inc]
                mini_batch_Y = shuffled_Y[:,k*inc:(k+1)*inc]
                # YOUR CODE ENDS HERE
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
            if m % mini_batch_size != 0:
                #(approx. 2 lines)
                # mini_batch_X =
                # mini_batch_Y =
                # YOUR CODE STARTS HERE
                mini_batch_X = shuffled_X[:,inc*num_complete_minibatches:]
                mini_batch_Y = shuffled_Y[:,inc*num_complete_minibatches:]  
                # YOUR CODE ENDS HERE
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            return mini_batches
        ```

        ```python
        t_X, t_Y, mini_batch_size = random_mini_batches_test_case()
        mini_batches = random_mini_batches(t_X, t_Y, mini_batch_size)

        print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
        print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
        print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
        print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
        print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
        print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
        print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

        random_mini_batches_test(random_mini_batches)

        '''
        shape of the 1st mini_batch_X: (12288, 64)
        shape of the 2nd mini_batch_X: (12288, 64)
        shape of the 3rd mini_batch_X: (12288, 20)
        shape of the 1st mini_batch_Y: (1, 64)
        shape of the 2nd mini_batch_Y: (1, 64)
        shape of the 3rd mini_batch_Y: (1, 20)
        mini batch sanity check: [ 0.90085595 -0.7612069   0.2344157 ]
        '''
        ```

Momentum

- because mini-batch GD makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch GD will “oscillate” toward convergence
- using momentum can reduce the oscillations
- momentum takes into account the past gradients to smooth out the update
- the direction of the previous gradient is stored in the variable v
- formally, this will be the exponentially weighted average of the gradient on previous steps
- think of v as the velocity of a ball rolling downhill, building up speed to the direction of the gradient/slope of the hill
- Exercise 3 - initialize_velocity
    - initialize the velocity
    - the velocity, v, is a python dictionary that needs to be initialized with arrays of zeros
    - its keys are the same as those in the grads dictionary

        ```python
        v["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
        v["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])
        ```

    - the iterator l starts at 1 in the for loop as the first parameters are v[”dW1”] and v[”db1”]

    ```python
    # GRADED FUNCTION: initialize_velocity

    def initialize_velocity(parameters):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl

        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """

        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}

        # Initialize velocity
        for l in range(1, L + 1):
            # (approx. 2 lines)
            # v["dW" + str(l)] =
            # v["db" + str(l)] =
            # YOUR CODE STARTS HERE'
            v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
            v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
            # YOUR CODE ENDS HERE

        return v

    '''
    v["dW1"] =
    [[0. 0.]
     [0. 0.]
     [0. 0.]]
    v["db1"] =
    [[0.]
     [0.]
     [0.]]
    v["dW2"] =
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    v["db2"] =
    [[0.]
     [0.]
     [0.]]
    '''
    ```

- Exercise 4 - update_paramters_with_momentum
    - implement the parameters update with momentum
    - the momentum update rule is, for l = 1, …, L

    - where L is the number of layers, beta is the momentum and alpha is the learning rate
    - all parameters should be stored in the parameters dictionary
    - note that the iterator l starts at 1 in the for loop as the first parameters are W[1] and b[1]

    ```python
    # GRADED FUNCTION: update_parameters_with_momentum

    def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
        """
        Update parameters using Momentum

        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- python dictionary containing the current velocity:
                        v['dW' + str(l)] = ...
                        v['db' + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        learning_rate -- the learning rate, scalar

        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- python dictionary containing your updated velocities
        """

        L = len(parameters) // 2 # number of layers in the neural networks

        # Momentum update for each parameter
        for l in range(1, L + 1):

            # (approx. 4 lines)
            # compute velocities
            # v["dW" + str(l)] = ...
            # v["db" + str(l)] = ...
            # update parameters
            # parameters["W" + str(l)] = ...
            # parameters["b" + str(l)] = ...
            # YOUR CODE STARTS HERE
            v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
            v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
            parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
            parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]
            # YOUR CODE ENDS HERE

        return parameters, v

    '''
    W1 = 
    [[ 1.62522322 -0.61179863 -0.52875457]
     [-1.071868    0.86426291 -2.30244029]]
    b1 = 
    [[ 1.74430927]
     [-0.76210776]]
    W2 = 
    [[ 0.31972282 -0.24924749]
     [ 1.46304371 -2.05987282]
     [-0.32294756 -0.38336269]]
    b2 = 
    [[ 1.1341662 ]
     [-1.09920409]
     [-0.171583  ]]
    v["dW1"] = 
    [[-0.08778584  0.00422137  0.05828152]
     [-0.11006192  0.11447237  0.09015907]]
    v["db1"] = 
    [[0.05024943]
     [0.09008559]]
    v["dW2"] = 
    [[-0.06837279 -0.01228902]
     [-0.09357694 -0.02678881]
     [ 0.05303555 -0.06916608]]
    v["db2"] = v[[-0.03967535]
     [-0.06871727]
     [-0.08452056]]
    '''
    ```

    - the velocity is initialized with zeros, so the alg will take a few iterations to build up velocity and start to take bigger steps
    - if beta = 0, then this just becomes standard GD without momentum
    - the larger the momentum beta is, the smoother the update, because it takes the past gradients into account more
    - if beta is too big, it could also smoot out the updates too much
    - common values for beta range from 0.8 to 0.999
    - don’t feel inclined to tune this, beta = 0.9 is fine
    - tuning the optimal beta for your model might require trying several values to see what works best in terms of reducing the value of the cost function J

Adam

- one of the most effective optimization algorithms for training NN
- it combines ideas from RMSProp and Momentum
    - it calculates an exponentially weighted average of past gradients, and stores it in variables v (before bias correction) and vcorrected (with bias correction)
    - it calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables s (before bias correction) and scorrected (with bias correction)
    - it updates parameters in a direction based on combining information from the previous two steps
- the update rule if, for l = 1, .., L

- where
    - t counts the number of steps taken of Adam
    - L is the number of layers
    - beta1 and beta2 are hyperparameters that control the two exponentially weighted averages
    - alpha is the learning rate
    - epsilon is a very small number to avoid dividing by zero
- Exercise 5 - initialize_adam
    - initialize the adam variables c, s which keep track of the past information
    - the variables v, s are python dictionaries that need to be initialized with arrays of zeros
    - their keys are the same for grads, that is: for l = 1, …, L

    ```python
    v["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
    v["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])
    s["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
    s["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])

    ```

    ```python
    # GRADED FUNCTION: initialize_adam

    def initialize_adam(parameters) :
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters["W" + str(l)] = Wl
                        parameters["b" + str(l)] = bl

        Returns: 
        v -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                        v["dW" + str(l)] = ...
                        v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                        s["dW" + str(l)] = ...
                        s["db" + str(l)] = ...

        """

        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}

        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(1, L + 1):
        # (approx. 4 lines)
            # v["dW" + str(l)] = ...
            # v["db" + str(l)] = ...
            # s["dW" + str(l)] = ...
            # s["db" + str(l)] = ...
        # YOUR CODE STARTS HERE
            v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
            v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
            s["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
            s["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
        # YOUR CODE ENDS HERE

        return v, s

    '''
    v["dW1"] = 
    [[0. 0. 0.]
     [0. 0. 0.]]
    v["db1"] = 
    [[0.]
     [0.]]
    v["dW2"] = 
    [[0. 0.]
     [0. 0.]
     [0. 0.]]
    v["db2"] = 
    [[0.]
     [0.]
     [0.]]
    s["dW1"] = 
    [[0. 0. 0.]
     [0. 0. 0.]]
    s["db1"] = 
    [[0.]
     [0.]]
    s["dW2"] = 
    [[0. 0.]
     [0. 0.]
     [0. 0.]]
    s["db2"] = 
    [[0.]
     [0.]
     [0.]]
    '''
    ```

- Exercise 6 - update_parameters_with_adam
    - implement the parameters update with Adam

    ```python
    # GRADED FUNCTION: update_parameters_with_adam

    def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                    beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        """
        Update parameters using the Adam optimization algorithm.

        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameter:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, maintained as a python dictionary
        s -- Adam variable, moving average of the squared gradient, maintained as a python dictionary
        t -- Adam variable, counts the number of steps taken for bias correction
        learning_rate -- learning rate, scalar
        beta1 -- exponential decay hyperparameter for the first moment estimates
        beta2 -- exponential decay hyperparameter for the second moment estimates
        epsilon -- hyperparameter to prevent division by zero in the Adam update, a small scalar

        Returns:
        parameters -- python dictionary containing your updated parameters
        v -- updated Adam variable, moving average of the first gradient, python dictionary
        s -- updated Adam variable, moving average of the squared gradient, python dictionary
        v_corrected -- python dictionary containing bias-corrected first moment estimates
        s_corrected -- python dictionary containing bias-corrected second moment estimates
        """

        L = len(parameters) // 2                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        for l in range(1, L + 1):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            # (approx. 2 lines)
            # v["dW" + str(l)] = ...
            # v["db" + str(l)] = ...
            # YOUR CODE STARTS HERE
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
            # YOUR CODE ENDS HERE

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            # (approx. 2 lines)
            # v_corrected["dW" + str(l)] = ...
            # v_corrected["db" + str(l)] = ...
            # YOUR CODE STARTS HERE
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
            # YOUR CODE ENDS HERE

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            #(approx. 2 lines)
            # s["dW" + str(l)] = ...
            # s["db" + str(l)] = ...
            # YOUR CODE STARTS HERE
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)], 2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)], 2)  
            # YOUR CODE ENDS HERE

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            # (approx. 2 lines)
            # s_corrected["dW" + str(l)] = ...
            # s_corrected["db" + str(l)] = ...
            # YOUR CODE STARTS HERE
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))
            # YOUR CODE ENDS HERE

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            # (approx. 2 lines)
            # parameters["W" + str(l)] = ...
            # parameters["b" + str(l)] = ...
            # YOUR CODE STARTS HERE
            parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
            parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
            # YOUR CODE ENDS HERE

        return parameters, v, s, v_corrected, s_corrected

    '''
    W1 = 
    [[ 1.63937725 -0.62327448 -0.54308727]
     [-1.0578897   0.85032154 -2.31657668]]
    W2 = 
    [[ 0.33400549 -0.23563857]
     [ 1.47715417 -2.04561842]
     [-0.33729882 -0.36908457]]
    b1 = 
    [[ 1.72995096]
     [-0.7762447 ]]
    b2 = 
    [[ 1.14852557]
     [-1.08492339]
     [-0.15740527]]
    '''
    ```

Model with different Optimization algorithms

- use the following moons dataset to test the different optimization methods

    ```python
    train_X, train_Y = load_dataset()
    ```

- a three layer NN
    - mini-batch GD will call: update_parameters_with_gd()
    - mini-batch momentum will call: initialize_velocity() and update_parameters_with_momentum()
    - mini_batch Adam will call: initialize_adam() and update_parameters_with_adam()

    ```python
    def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
              beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True):
        """
        3-layer neural network model which can be run in different optimizer modes.

        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        optimizer -- the optimizer to be passed, gradient descent, momentum or adam
        layers_dims -- python list, containing the size of each layer
        learning_rate -- the learning rate, scalar.
        mini_batch_size -- the size of a mini batch
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates 
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates
        num_epochs -- number of epochs
        print_cost -- True to print the cost every 1000 epochs

        Returns:
        parameters -- python dictionary containing your updated parameters 
        """

        L = len(layers_dims)             # number of layers in the neural networks
        costs = []                       # to keep track of the cost
        t = 0                            # initializing the counter required for Adam update
        seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
        m = X.shape[1]                   # number of training examples

        # Initialize parameters
        parameters = initialize_parameters(layers_dims)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no initialization required for gradient descent
        elif optimizer == "momentum":
            v = initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = initialize_adam(parameters)

        # Optimization loop
        for i in range(num_epochs):

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
            cost_total = 0

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                a3, caches = forward_propagation(minibatch_X, parameters)

                # Compute cost and add to the cost total
                cost_total += compute_cost(a3, minibatch_Y)

                # Backward propagation
                grads = backward_propagation(minibatch_X, minibatch_Y, caches)

                # Update parameters
                if optimizer == "gd":
                    parameters = update_parameters_with_gd(parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                                   t, learning_rate, beta1, beta2,  epsilon)
            cost_avg = cost_total / m

            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            if print_cost and i % 100 == 0:
                costs.append(cost_avg)

        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        return parameters
    ```

- Mini-Batch GD
    - run with min-batch GD

    ```python
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    '''
    Cost after epoch 0: 0.702405
    Cost after epoch 1000: 0.668101
    Cost after epoch 2000: 0.635288
    Cost after epoch 3000: 0.600491
    Cost after epoch 4000: 0.573367
    '''
    ```

- Mini-Batch GD with Momentum
    - run the model with momentum
    - gains from simple example are small, greater with more complex problems

    ```python
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Momentum optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    '''
    Cost after epoch 0: 0.702413
    Cost after epoch 1000: 0.668167
    Cost after epoch 2000: 0.635388
    Cost after epoch 3000: 0.600591
    Cost after epoch 4000: 0.573444
    '''
    ```

- Mini-Batch with Adam
    - run the model with Adam

    ```python
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    '''
    Cost after epoch 0: 0.702166
    Cost after epoch 1000: 0.167845
    Cost after epoch 2000: 0.141316
    Cost after epoch 3000: 0.138788
    Cost after epoch 4000: 0.136066
    '''
    ```

- Summary
    - momentum usually helps, but given the small learning rate and simplistic dataset, its impact is almost negligible
    - adam clearly outperforms mini-batch GD and momentum
    - if you run the model for more epochs on the dataset, all three methods will lead to good results, however adam converges much faster
    - advantages of adam include
        - relatively low memory requirements (though higher than GD and GD with momentum)
        - usually works well even with little tuning of hyperparameters (except alpha)

Learning Rate Decay and Scheduling

- during the first part of training, the model can get away with taking large steps, but over time, using a fixed value for the learning rate alpha can cause the model to get stuck in a wide oscillation that never quite converges
- slowly reducing the learning rate alpha over time, small and slower steps can be taken to get closer to the minimum
- can be achieved by using either adaptive methods or pre-defined learning rate schedules
- apply learning rate decay on a three layer NN in three different optimizer modes and se how each one differs, as well as the effect of scheduling at different epochs

    ```python
    def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
              beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True, decay=None, decay_rate=1):
        """
        3-layer neural network model which can be run in different optimizer modes.

        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        layers_dims -- python list, containing the size of each layer
        learning_rate -- the learning rate, scalar.
        mini_batch_size -- the size of a mini batch
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates 
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates
        num_epochs -- number of epochs
        print_cost -- True to print the cost every 1000 epochs

        Returns:
        parameters -- python dictionary containing your updated parameters 
        """

        L = len(layers_dims)             # number of layers in the neural networks
        costs = []                       # to keep track of the cost
        t = 0                            # initializing the counter required for Adam update
        seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
        m = X.shape[1]                   # number of training examples
        lr_rates = []
        learning_rate0 = learning_rate   # the original learning rate

        # Initialize parameters
        parameters = initialize_parameters(layers_dims)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no initialization required for gradient descent
        elif optimizer == "momentum":
            v = initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = initialize_adam(parameters)

        # Optimization loop
        for i in range(num_epochs):

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
            cost_total = 0

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                a3, caches = forward_propagation(minibatch_X, parameters)

                # Compute cost and add to the cost total
                cost_total += compute_cost(a3, minibatch_Y)

                # Backward propagation
                grads = backward_propagation(minibatch_X, minibatch_Y, caches)

                # Update parameters
                if optimizer == "gd":
                    parameters = update_parameters_with_gd(parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                                   t, learning_rate, beta1, beta2,  epsilon)
            cost_avg = cost_total / m
            if decay:
                learning_rate = decay(learning_rate0, i, decay_rate)
            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
                if decay:
                    print("learning rate after epoch %i: %f"%(i, learning_rate))
            if print_cost and i % 100 == 0:
                costs.append(cost_avg)

        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        return parameters
    ```

- Decay on every iteration
    - try one of the pre-defined schedules for learning rate decay, called exponential learning rate decay

    - Exercise 7 - update_lr

        ```python
        # GRADED FUNCTION: update_lr

        def update_lr(learning_rate0, epoch_num, decay_rate):
            """
            Calculates updated the learning rate using exponential weight decay.

            Arguments:
            learning_rate0 -- Original learning rate. Scalar
            epoch_num -- Epoch number. Integer
            decay_rate -- Decay rate. Scalar

            Returns:
            learning_rate -- Updated learning rate. Scalar 
            """
            #(approx. 1 line)
            # learning_rate = 
            # YOUR CODE STARTS HERE
            learning_rate = 1 / (1 + decay_rate * epoch_num) * learning_rate0    
            # YOUR CODE ENDS HERE
            return learning_rate
        ```

        ```python
        # train 3-layer model
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters = model(train_X, train_Y, layers_dims, optimizer = "gd", learning_rate = 0.1, num_epochs=5000, decay=update_lr)

        # Predict
        predictions = predict(train_X, train_Y, parameters)

        # Plot decision boundary
        plt.title("Model with Gradient Descent optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])
        plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

        '''
        Cost after epoch 0: 0.701091
        learning rate after epoch 0: 0.100000
        Cost after epoch 1000: 0.661884
        learning rate after epoch 1000: 0.000100
        Cost after epoch 2000: 0.658620
        learning rate after epoch 2000: 0.000050
        Cost after epoch 3000: 0.656765
        learning rate after epoch 3000: 0.000033
        Cost after epoch 4000: 0.655486
        learning rate after epoch 4000: 0.000025
        '''
        ```

    - if decay occurs every iteration, learning rate goes to zero too quickly

    - running a few epochs doesn’t cause a lot of troubles
- Fixed Interval Scheduling
    - help prevent the learning rate speeding to zero too quickly by scheduling the exponential learning rate decay at a fixed time interval
    - either number the intervals, or divide the epoch by the time interval, which is the size of window with the constant learning rate

    - Exercise 8 - schedule_lr_decay
        - calculating the new learning rate using exponential weight decay with fixed interval scheduling
        - implement the learning rate scheduling such that it only changes when the epochNum is a multiple of the timeInterval
        - the fraction in the denominator uses the floor operation

        ```python
        # GRADED FUNCTION: schedule_lr_decay

        def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
            """
            Calculates updated the learning rate using exponential weight decay.

            Arguments:
            learning_rate0 -- Original learning rate. Scalar
            epoch_num -- Epoch number. Integer.
            decay_rate -- Decay rate. Scalar.
            time_interval -- Number of epochs where you update the learning rate.

            Returns:
            learning_rate -- Updated learning rate. Scalar 
            """
            # (approx. 1 lines)
            # learning_rate = ...
            # YOUR CODE STARTS HERE
            learning_rate = 1 / (1 + decay_rate * np.floor(epoch_num/time_interval)) * learning_rate0    
            # YOUR CODE ENDS HERE
            return learning_rate

        '''
        Original learning rate:  0.5
        Updated learning rate after 10 epochs:  0.5
        Updated learning rate after 100 epochs:  0.3846153846153846
        '''
        ```

- Using Learning Rate Decay for each Optimization Method
    - use the moons dataset to test the different optimization methods
    - Gradient Descent with Learning Rate Decay
        - run the model with GD and learning rate decay

            ```python
            # train 3-layer model
            layers_dims = [train_X.shape[0], 5, 2, 1]
            parameters = model(train_X, train_Y, layers_dims, optimizer = "gd", learning_rate = 0.1, num_epochs=5000, decay=schedule_lr_decay)

            # Predict
            predictions = predict(train_X, train_Y, parameters)

            # Plot decision boundary
            plt.title("Model with Gradient Descent optimization")
            axes = plt.gca()
            axes.set_xlim([-1.5,2.5])
            axes.set_ylim([-1,1.5])
            plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

            '''
            Cost after epoch 0: 0.701091
            learning rate after epoch 0: 0.100000
            Cost after epoch 1000: 0.127161
            learning rate after epoch 1000: 0.050000
            Cost after epoch 2000: 0.120304
            learning rate after epoch 2000: 0.033333
            Cost after epoch 3000: 0.117033
            learning rate after epoch 3000: 0.025000
            Cost after epoch 4000: 0.117512
            learning rate after epoch 4000: 0.020000
            '''
            ```

    - Gradient Descent with Momentum and Learning Rate Decay
        - run the model with GD with momentum and learning rate decay

            ```python
            # train 3-layer model
            layers_dims = [train_X.shape[0], 5, 2, 1]
            parameters = model(train_X, train_Y, layers_dims, optimizer = "momentum", learning_rate = 0.1, num_epochs=5000, decay=schedule_lr_decay)

            # Predict
            predictions = predict(train_X, train_Y, parameters)

            # Plot decision boundary
            plt.title("Model with Gradient Descent with momentum optimization")
            axes = plt.gca()
            axes.set_xlim([-1.5,2.5])
            axes.set_ylim([-1,1.5])
            plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

            '''
            Cost after epoch 0: 0.702226
            learning rate after epoch 0: 0.100000
            Cost after epoch 1000: 0.128974
            learning rate after epoch 1000: 0.050000
            Cost after epoch 2000: 0.125965
            learning rate after epoch 2000: 0.033333
            Cost after epoch 3000: 0.123375
            learning rate after epoch 3000: 0.025000
            Cost after epoch 4000: 0.123218
            learning rate after epoch 4000: 0.020000
            '''
            ```

    - Adam with Learning Rate Decay
        - run the model with adam and learning rate decay

            ```python
            # train 3-layer model
            layers_dims = [train_X.shape[0], 5, 2, 1]
            parameters = model(train_X, train_Y, layers_dims, optimizer = "adam", learning_rate = 0.01, num_epochs=5000, decay=schedule_lr_decay)

            # Predict
            predictions = predict(train_X, train_Y, parameters)

            # Plot decision boundary
            plt.title("Model with Adam optimization")
            axes = plt.gca()
            axes.set_xlim([-1.5,2.5])
            axes.set_ylim([-1,1.5])
            plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

            '''
            Cost after epoch 0: 0.699346
            learning rate after epoch 0: 0.010000
            Cost after epoch 1000: 0.130074
            learning rate after epoch 1000: 0.005000
            Cost after epoch 2000: 0.129826
            learning rate after epoch 2000: 0.003333
            Cost after epoch 3000: 0.129282
            learning rate after epoch 3000: 0.002500
            Cost after epoch 4000: 0.128361
            learning rate after epoch 4000: 0.002000
            '''
            ```

- Achieving similar performance with different methods
    - with mini-batch GD or mini-batch GD with momentum, the accuracy is significantly lower than adam, but when learning rate decay is added on top, either can achieve performance at a speed and accuracy score that’s similar to adam
    - in the case of adam, notice that the learning curve achieves a similar accuracy but faster