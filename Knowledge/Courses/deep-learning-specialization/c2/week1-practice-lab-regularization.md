Regularization

- deep learning models have so much flexibility and capacity that overfitting can be a serious problem, it training dataset is not big enough
- the learned network doesn’t generalize to new examples that is has never seen

Packages

```python
# import packages
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
from testCases import *
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2
```

Problem Statement

- you have been hired as an ai expert by the French football corporation
- they would like you to recommend positions where France’s goal keeper should kick the ball so that the French team’s players can then hit it with their head

- they give you the following 2D dataset from France’s past 10 games

Loading the Dataset

```python
train_X, train_Y, test_X, test_Y = load_2D_dataset()
```

- each dot corresponds to a position on the football field where a football player has hit the ball with hi/her head after the French goal keeper has shot the ball from the left side of the football field
    - if the dot is blue, it means the French player managed to hit the ball with his/her head
    - if the dot is red, it means the other team’s player hit the ball with their head
- goal is to use a deep learning model to find the position on the field where teh goalkeeper should kick the ball
- analysis of the dataset
    - the dataset is a little noisy
    - a diagonal line separating the upper left half from the lower right half would work well
    - first try a non-regularized model
    - then regularize it

Non-Regularized Model

- use the following NN
    - in regularization mode → by setting the lambd input to a non-zero value
    - in dropout mode → by setting the keep_prob to a value less than one
- first try the model without any regularization
    - L2 regularization → functions “compute_cost_with_regularization()” and “backward_propagation_with_regualrization()”
    - Dropout → functions: “forward_propagation_with_dropout()” and “backward_propagation_with_dropout()”
- run the model with the correct inputs so that it calls the functions implemented

    ```python
    def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
        """
        Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
        learning_rate -- learning rate of the optimization
        num_iterations -- number of iterations of the optimization loop
        print_cost -- If True, print the cost every 10000 iterations
        lambd -- regularization hyperparameter, scalar
        keep_prob - probability of keeping a neuron active during drop-out, scalar.

        Returns:
        parameters -- parameters learned by the model. They can then be used to predict.
        """

        grads = {}
        costs = []                            # to keep track of the cost
        m = X.shape[1]                        # number of examples
        layers_dims = [X.shape[0], 20, 3, 1]

        # Initialize parameters dictionary.
        parameters = initialize_parameters(layers_dims)

        # Loop (gradient descent)

        for i in range(0, num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            if keep_prob == 1:
                a3, cache = forward_propagation(X, parameters)
            elif keep_prob < 1:
                a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

            # Cost function
            if lambd == 0:
                cost = compute_cost(a3, Y)
            else:
                cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

            # Backward propagation.
            assert (lambd == 0 or keep_prob == 1)   # it is possible to use both L2 regularization and dropout, 
                                                    # but this assignment will only explore one at a time
            if lambd == 0 and keep_prob == 1:
                grads = backward_propagation(X, Y, cache)
            elif lambd != 0:
                grads = backward_propagation_with_regularization(X, Y, cache, lambd)
            elif keep_prob < 1:
                grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

            # Print the loss every 10000 iterations
            if print_cost and i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
            if print_cost and i % 1000 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return parameters
    ```

- train the model without any regularization

    ```python
    parameters = model(train_X, train_Y)
    print ("On the training set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    '''
    Cost after iteration 0: 0.6557412523481002
    Cost after iteration 10000: 0.16329987525724204
    Cost after iteration 20000: 0.13851642423234922

    On the training set:
    Accuracy: 0.9478672985781991
    On the test set:
    Accuracy: 0.915
    '''
    ```

- the train accuracy is 94.8% while the test accuracy is 91.5%
- this is the baseline model
- plot the decision boundary

    ```python
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    ```

- the non-regularized model is obviously overfitting the training set
- it is fitting the noisy points

L2 Regularization

- the standard way to avoid overfitting is called L2 regularization
- it consist of appropriately modifying the cost function

- Exercise 1 - compute_cost_with_regularization
    - implement compute_cost_with_regularization() which compute the cost
    - to calculate the sums use np.sum(np.square(Wl))
    - do this for W[1], W[2], and W[3], then sum the three terms and multiply by 1/m * lambda/2

    ```python
    # GRADED FUNCTION: compute_cost_with_regularization

    def compute_cost_with_regularization(A3, Y, parameters, lambd):
        """
        Implement the cost function with L2 regularization. See formula (2) above.

        Arguments:
        A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model

        Returns:
        cost - value of the regularized loss function (formula (2))
        """
        m = Y.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        W3 = parameters["W3"]

        cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost

        #(≈ 1 lines of code)
        # L2_regularization_cost = 
        # YOUR CODE STARTS HERE
        W1_sum = np.sum(np.square(W1))
        W2_sum = np.sum(np.square(W2))
        W3_sum = np.sum(np.square(W3))
        L2_regularization_cost = ((1/m) * (lambd/2) * (W1_sum + W2_sum + W3_sum))
        # YOUR CODE ENDS HERE

        cost = cross_entropy_cost + L2_regularization_cost

        return cost
    ```

- since the cost was changes, the backward prop has to change as well
- all the gradients have to be computed w.r.t the new cost
- Exercise 2 - backward_propatation_with_regularization
    - implement the changes needed in back prop to take into account regularization
    - the changes only concern dW1, dW2 and dW3
    - for each, add the regularization term’s gradient lambda/m * W

    ```python
    # GRADED FUNCTION: backward_propagation_with_regularization

    def backward_propagation_with_regularization(X, Y, cache, lambd):
        """
        Implements the backward propagation of our baseline model to which we added an L2 regularization.

        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar

        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """

        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

        dZ3 = A3 - Y
        #(≈ 1 lines of code)
        # dW3 = 1./m * np.dot(dZ3, A2.T) + None
        # YOUR CODE STARTS HERE
        dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m * W3)  
        # YOUR CODE ENDS HERE
        db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        #(≈ 1 lines of code)
        # dW2 = 1./m * np.dot(dZ2, A1.T) + None
        # YOUR CODE STARTS HERE
        dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m * W2)    
        # YOUR CODE ENDS HERE
        db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        #(≈ 1 lines of code)
        # dW1 = 1./m * np.dot(dZ1, X.T) + None
        # YOUR CODE STARTS HERE
        dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m * W1) 
        # YOUR CODE ENDS HERE
        db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                     "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                     "dZ1": dZ1, "dW1": dW1, "db1": db1}

        return gradients

    '''
    dW1 = 
    [[-0.25604646  0.12298827 -0.28297129]
     [-0.17706303  0.34536094 -0.4410571 ]]
    dW2 = 
    [[ 0.79276486  0.85133918]
     [-0.0957219  -0.01720463]
     [-0.13100772 -0.03750433]]
    dW3 = 
    [[-1.77691347 -0.11832879 -0.09397446]]   

    '''
    ```

- run the model with L2 regularization (lambda = 0.7)
- the model function will call
    - compute_cost_with_regualrization instead of compute_cost
    - backward_propagation_with_regualrization instead of backward_propagation

    ```python
    parameters = model(train_X, train_Y, lambd = 0.7)
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    '''
    Cost after iteration 0: 0.6974484493131264
    Cost after iteration 10000: 0.2684918873282238
    Cost after iteration 20000: 0.26809163371273004

    On the train set:
    Accuracy: 0.9383886255924171
    On the test set:
    Accuracy: 0.93
    '''
    ```

- the test set accuracy increase to 93%
- not overfitting the training anymore
- plot decision boundary

    ```python
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    ```

- the value of lambda is a hyperparameter that can be tuned
- L2 regularization makes the decision boundary smoother
- if lambda is too large, it is also possible to “oversmooth”, resulting in a model with high bias
- What is L2-regularization actually doing?
    - L2-regularization relies on the assumption that  model with small weights is simpler than a model with large weights
    - thus, by penalizing the square values of the weights in the cost function, you drive all the weights to smaller values
    - it becomes too costly for the cost to have large weights
    - this leads to a smoother model in which the output changes more slowly as the input changes

Dropout

- dropout is a widely used regularization technique that is specific to deep learning
- it randomly shuts down some neurons in each iteration
- at each iteration, shut down each neuron of a layer with probability 1 - keep_prob or keep it with the probability keep_prob
- the dropped neurons don’t contribute to the training in both the forward and backward props of the iteration
- shutting down some neurons actually modifies the model
- the idea behind drop-out is that at each iteration, train a different model that uses only a subset of the neurons
- with dropout, the neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time
- Forward Propagation with Dropout
    - Exercise 3 - forward_propagation_with_dropout
        - implement the forward prop with dropout
        - using a 3 layer NN, and will add dropout to the first and second hidden layers
        - no dropout applied to the input or output layer
            - create a variable d[l] with the same shape as a[l] using np.random.rand() to randomly get number between 0 and 1
            - use a vectorized implementation, so create a random matrix D[l] of the same dimension as A[l]
            - set each entry of D[l] to be 1 with prob keep_prob, and 0 otherwise
        - if keep_prob is 0.8, which means that 80% of the neurons will be kept and 20 of them will be dropped
        - generate a vector that has 1’s and 0’s, where about 80% orf them are 1 and about 20% are 0: X = (X < keep_prob).astype(int) is the same as

        - note that the line works with multi-dimensional arrays, and the resulting output preserves the dimensions of the input array
        - without useing .astype(int), the result is an array of booleans True and False, which Python automatically converts to 1 and 0 when multiplied
        - however, it’s better practice to convert data into the data type that is intended
            - set A[l] to A[l] * D[l], think of D[l] as a mask, so that when it is multiplied with another matrix, it shuts down some of the values
            - divide A[l] by keep_prob, assuring that the result of the cost will still have the same expected value as without drop-out (inverse dropout)

        ```python
        # GRADED FUNCTION: forward_propagation_with_dropout

        def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
            """
            Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

            Arguments:
            X -- input dataset, of shape (2, number of examples)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                            W1 -- weight matrix of shape (20, 2)
                            b1 -- bias vector of shape (20, 1)
                            W2 -- weight matrix of shape (3, 20)
                            b2 -- bias vector of shape (3, 1)
                            W3 -- weight matrix of shape (1, 3)
                            b3 -- bias vector of shape (1, 1)
            keep_prob - probability of keeping a neuron active during drop-out, scalar

            Returns:
            A3 -- last activation value, output of the forward propagation, of shape (1,1)
            cache -- tuple, information stored for computing the backward propagation
            """

            np.random.seed(1)

            # retrieve parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]

            # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
            Z1 = np.dot(W1, X) + b1
            A1 = relu(Z1)
            #(≈ 4 lines of code)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
            # D1 =                                           # Step 1: initialize matrix D1 = np.random.rand(..., ...)
            # D1 =                                           # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
            # A1 =                                           # Step 3: shut down some neurons of A1
            # A1 =                                           # Step 4: scale the value of neurons that haven't been shut down
            # YOUR CODE STARTS HERE
            D1 = np.random.rand(A1.shape[0], A1.shape[1])
            D1 = (D1 < keep_prob).astype(int)
            A1 *= D1
            A1 /= keep_prob
            # YOUR CODE ENDS HERE
            Z2 = np.dot(W2, A1) + b2
            A2 = relu(Z2)
            #(≈ 4 lines of code)
            # D2 =                                           # Step 1: initialize matrix D2 = np.random.rand(..., ...)
            # D2 =                                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
            # A2 =                                           # Step 3: shut down some neurons of A2
            # A2 =                                           # Step 4: scale the value of neurons that haven't been shut down
            # YOUR CODE STARTS HERE
            D2 = np.random.rand(A2.shape[0], A2.shape[1])
            D2 = (D2 < keep_prob).astype(int)
            A2 *= D2
            A2 /= keep_prob
            # YOUR CODE ENDS HERE
            Z3 = np.dot(W3, A2) + b3
            A3 = sigmoid(Z3)

            cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

            return A3, cache
        ```

- Backward Prop with Dropout
    - Exercise 4 - backward_propagation_with_dropout
        - implement the backward propagation with dropout
        - add dropout to the first and second hidden layers, using the masks D[1] and D[2] stored in the cache
        - backpropagation with dropout is quite easy
            - previously shut down neurons with forward prop, by applying a mask D[1] to A1
            - in backprop, shut down the same neurons, by reapplying the same mask D[1] to dA[1]
            - during forward prop, A1 was divided by keep_prob
            - in backptop, divide by dA[1] by keep_prob again
            - the calculus interpretation is that if A[1] is scaled by keep_prob, then its derivative dA[1] is also scaled by the same keep_prob)

        ```python
        # GRADED FUNCTION: backward_propagation_with_dropout

        def backward_propagation_with_dropout(X, Y, cache, keep_prob):
            """
            Implements the backward propagation of our baseline model to which we added dropout.

            Arguments:
            X -- input dataset, of shape (2, number of examples)
            Y -- "true" labels vector, of shape (output size, number of examples)
            cache -- cache output from forward_propagation_with_dropout()
            keep_prob - probability of keeping a neuron active during drop-out, scalar

            Returns:
            gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
            """

            m = X.shape[1]
            (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

            dZ3 = A3 - Y
            dW3 = 1./m * np.dot(dZ3, A2.T)
            db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
            dA2 = np.dot(W3.T, dZ3)
            #(≈ 2 lines of code)
            # dA2 =                # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
            # dA2 =                # Step 2: Scale the value of neurons that haven't been shut down
            # YOUR CODE STARTS HERE
            dA2 *= D2
            dA2 /= keep_prob
            # YOUR CODE ENDS HERE
            dZ2 = np.multiply(dA2, np.int64(A2 > 0))
            dW2 = 1./m * np.dot(dZ2, A1.T)
            db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.dot(W2.T, dZ2)
            #(≈ 2 lines of code)
            # dA1 =                # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
            # dA1 =                # Step 2: Scale the value of neurons that haven't been shut down
            # YOUR CODE STARTS HERE
            dA1 *= D1
            dA1 /= keep_prob    
            # YOUR CODE ENDS HERE
            dZ1 = np.multiply(dA1, np.int64(A1 > 0))
            dW1 = 1./m * np.dot(dZ1, X.T)
            db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

            gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                         "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                         "dZ1": dZ1, "dW1": dW1, "db1": db1}

            return gradients
        ```

        - run the model with dropout keep_prob = 0.86
        - at every iteration shut down each neurons of layer 1 and 2 with 14% prob
        - the function model() will now call
            - forward_propagation_with_dropout instead of forward_propagation
            - backward_propagation_with_dropout instead of backward_propagation

        ```python
        parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

        print ("On the train set:")
        predictions_train = predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = predict(test_X, test_Y, parameters)

        '''
        Cost after iteration 0: 0.6543912405149825
        Cost after iteration 10000: 0.0610169865749056
        Cost after iteration 20000: 0.060582435798513114

        On the train set:
        Accuracy: 0.9289099526066351
        On the test set:
        Accuracy: 0.95
        '''
        ```

        - the test accuracy has increase again to 95%
        - the model is not overfitting the training set and does a great job on the test set

        ```python
        plt.title("Model with dropout")
        axes = plt.gca()
        axes.set_xlim([-0.75,0.40])
        axes.set_ylim([-0.75,0.65])
        plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
        ```

        - a common mistake when using dropout is to use it both in training and testing
        - use dropout only in training
        - deeplearning frameworks come with a dropout layer implementation

Conclusions

- results of the three models

- regularization hurts training set performance
- this is because it limits the ability of the network to overfit to the training set
- but since it gives better test accuracy, it help the system