Initialization

- specify an initial value of the weights
- a well-chosen initialization can
    - speed up the convergence of gradient descent
    - increase the odds of GD converging to a lower training error

Packages

```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from public_tests import *
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

# load image dataset: blue/red dots in circles
# train_X, train_Y, test_X, test_Y = load_dataset()
```

Loading the Dataset

```python
train_X, train_Y, test_X, test_Y = load_dataset()
```

NN Model

- use a 3-layer NN
- zeros initialization → setting initialization = “zeros” in the input
- random initialization → setting initialization = “random” in the input argument, this initializes the weights to large random values
- HE initialization → setting initialization “he” in the input argument, this initializes the weights to random values scaled

```python
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    Returns:
    parameters -- parameters learnt by the model
    """

    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)

        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
```

Zero Initialization

- two types of parameters to initialize in a NN
    - the weight matrices
    - the bias vectors
- Exercise 1 - initialize_parameters_zeros
    - implement the function to initialize all parameters to zeros
    - fails to ‘break symmetry’

    ```python
    # GRADED FUNCTION: initialize_parameters_zeros 

    def initialize_parameters_zeros(layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """

        parameters = {}
        L = len(layers_dims)            # number of layers in the network

        for l in range(1, L):
            #(≈ 2 lines of code)
            # parameters['W' + str(l)] = 
            # parameters['b' + str(l)] = 
            # YOUR CODE STARTS HERE
            parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1])) 
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            # YOUR CODE ENDS HERE
        return parameters
    ```

- train the model on 15k iterations using zeros initialization

    ```python
    parameters = model(train_X, train_Y, initialization = "zeros")
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    '''
    Cost after iteration 0: 0.6931471805599453
    Cost after iteration 1000: 0.6931471805599453
    ...
    Cost after iteration 13000: 0.6931471805599453
    Cost after iteration 14000: 0.6931471805599453

    On the train set:
    Accuracy: 0.5
    On the test set:
    Accuracy: 0.5
    '''
    ```

- the performance is terrible, the cost doesn’t decrease, and the alg performs no better than random guessing
- look at the details of the predictions and the decision boundary

    ```python
    print ("predictions_train = " + str(predictions_train))
    print ("predictions_test = " + str(predictions_test))

    '''
    predictions_train = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ...
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0]]
    predictions_test = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

    '''
    ```

    ```python
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    ```

- comprehensive explanation from Paul Mielke’s post: [Symmetry Breaking versus Zero Initialization](https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061)
- simple explanation below, calculations below are done using only one example at a time
- the weights and biases are zero, multiplying by the weights creates the zero vector which gives zero when the activation function is ReLU, as z = 0

- at the classification layer, where the activation function is sigmoid

- as for every example, there is a 0.5 chance of it being true that the cost function becomes helpless in adjusting the weights
- with the loss function

- for y = 1, y_pred = 0.5, it becomes

- for y = 0, y_pred = 0.5, it becomes

- the prediction being 0.5 whether the actual value is 1 or 0 gives the same loss value for both, so none of the weights get adjusted and are stuck with the old value of the weights
- this is why the model is predicting 0 for every example
- in general, initializing the weights to zero results in the network fails to break symmetry
- every neuron in each layer will learn the same thing, might as well be training a NN with n[l] = 1 for every layer
- this way, the network is no more powerful than a linear classifier like logistic regression

Random Initialization

- to break symmetry, initialize the weights randomly
- each neuron can then proceed to learn a different function of its inputs
- Exercise 2 - initialize_parameters_random
    - implement the function to initialize the weights to large random values and the biases to zeros

    ```python
    # GRADED FUNCTION: initialize_parameters_random

    def initialize_parameters_random(layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """

        np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
        parameters = {}
        L = len(layers_dims)            # integer representing the number of layers

        for l in range(1, L):
            #(≈ 2 lines of code)
            # parameters['W' + str(l)] = 
            # parameters['b' + str(l)] =
            # YOUR CODE STARTS HERE
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            # YOUR CODE ENDS HERE

        return parameters

    '''
    W1 = [[ 17.88628473   4.36509851   0.96497468]
     [-18.63492703  -2.77388203  -3.54758979]]
    b1 = [[0.]
     [0.]]
    W2 = [[-0.82741481 -6.27000677]]
    b2 = [[0.]]
    '''
    ```

- train the model on 15k iteration using random initialization

    ```python
    parameters = model(train_X, train_Y, initialization = "random")
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    '''
    Cost after iteration 0: inf
    Cost after iteration 1000: 0.6247924745506072
    ...
    Cost after iteration 13000: 0.38497629552893475
    Cost after iteration 14000: 0.38276694641706693

    On the train set:
    Accuracy: 0.83
    On the test set:
    Accuracy: 0.86
    '''
    ```

- inf is because of numerical roundoff
- broken symmetry, gives better accuracy

    ```python
    print (predictions_train)
    print (predictions_test)

    '''
    [[1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1
      ...
      1 1 0 1 1 0 1 1 1 0 0 1 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1
      1 1 1 1 0 0 0 1 1 1 1 0]]
    [[1 1 1 1 0 1 0 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 0 1
      0 1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0
      1 1 1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0]]

    '''
    ```

    ```python
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    ```

- the cost starts very high
- this is because with large random-valued weights, the last activation outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example
- poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization alg
- training this network for longer would yield better results, but initializing with overly large random numbers slows down the optimization
- the main difference between Gaussian variable (np.random.randn())and uniform random variable is the distribution of the generated random numbers
    - np.random.rand() produces numbers in a uniform distribution
    - np.random.rand() produces numebr in a normal distribution
- when used for weight initialization, randn() helps most of the weights to avoid being close to the extremes, allocating most of them in the center of the range
- an intuitive way to see it is if you take the sigmoid activation function
- the slope near 0 or near 1 is extremely small, so the weights near those extremes will converge much more slowly to the solution, and having most of them near the center will speed the convergence

He Initialization

- names for the first author of He or similar “Xavier initialization”
- uses a scaling factor for the weights W[l] of sqrt(1./layers_dims[l-1] where He initialization would use sqrt(2./layers_dims[l-1])
- Exercise 3 - initialize_paramters_he
    - implement the function to initialize the parameters with He initialization
    - the function is similar to the previous initialize_parameters_random
    - difference is that instead of multiplying np.random.randn(..) by 10, multiply it by sqrt(2/dimension of the previous layer), which is what He initialization recommends for layers with a ReLU activation

    ```python
    # GRADED FUNCTION: initialize_parameters_he

    def initialize_parameters_he(layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1 # integer representing the number of layers

        for l in range(1, L + 1):
            #(≈ 2 lines of code)
            # parameters['W' + str(l)] = 
            # parameters['b' + str(l)] =
            # YOUR CODE STARTS HERE
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))        
            # YOUR CODE ENDS HERE

        return parameters

    '''
    W1 = [[ 1.78862847  0.43650985]
     [ 0.09649747 -1.8634927 ]
     [-0.2773882  -0.35475898]
     [-0.08274148 -0.62700068]]
    b1 = [[0.]
     [0.]
     [0.]
     [0.]]
    W2 = [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
    b2 = [[0.]]
    '''
    ```

- train the model

    ```python
    parameters = model(train_X, train_Y, initialization = "he")
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    '''
    Cost after iteration 0: 0.8830537463419761
    Cost after iteration 1000: 0.6879825919728063
    ...
    Cost after iteration 13000: 0.08457055954024283
    Cost after iteration 14000: 0.07357895962677366

    On the train set:
    Accuracy: 0.9933333333333333
    On the test set:
    Accuracy: 0.96
    '''
    ```

    ```python
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    ```

- the model separates the blue and the red dots very well

Conclusions

- three different types of initializations
- for the same number of iterations and same hyperparameters