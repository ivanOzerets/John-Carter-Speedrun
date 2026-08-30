Packages

- numpy
- sklearn - provides simple and efficient tools for data mining and data analysis
- matplotlib - library for plotting graphs in Python
- testCases provides some test examples to assess the correctness of the function
- planar_utils provide various useful functions

    ```python
    # Package imports
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    from testCases_v2 import *
    from public_tests import *
    import sklearn
    import sklearn.datasets
    import sklearn.linear_model
    from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

    %matplotlib inline

    %load_ext autoreload
    %autoreload 2
    ```

Load the Dataset

- loading

    ```python
    X, Y = load_planar_dataset()
    ```

- visualize the dataset using matplotlib
- the data looks like a ‘flower’ with some red and some blue points
- the goal is to build a model to fit the data
- need a classifier to define regions as either red or blue

    ```python
    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    ```

- have
    - numpy-array X that contains the features (x1, x2)
    - numpy-array Y that contains the labels
- Exercise 1
    - how many training examples are there
    - what is the shape of the variables X and Y

        ```python
        # (≈ 3 lines of code)
        # shape_X = ...
        # shape_Y = ...
        # training set size
        # m = ...
        # YOUR CODE STARTS HERE
        shape_X = X.shape
        shape_Y = Y.shape
        m = X.shape[1]
        # YOUR CODE ENDS HERE

        print ('The shape of X is: ' + str(shape_X))
        print ('The shape of Y is: ' + str(shape_Y))
        print ('I have m = %d training examples!' % (m))

        '''
        The shape of X is: (2, 400)
        The shape of Y is: (1, 400)
        I have m = 400 training examples!
        '''
        ```

Simple Logistic Regression

- use sklearn’s built-in function
- run the following code to train a logistic regression classifier on the dataset

    ```python
    # Train the logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV();
    clf.fit(X.T, Y.T); # why transpose?
    ```

- plot the decision boundary of the models

    ```python
    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
           '% ' + "(percentage of correctly labelled datapoints)")

    # Accuracy of logistic regression: 47 % (percentage of correctly labelled datapoints)
    ```

- interpretation: the dataset is not linearly separable, so logistic regression doesn’t perform well

NN model

- train a NN with a single hidden layer
- the model

- mathematically

- given the predictions on all the examples, compute the cost J

- reminder - the general methodology to build a NN
    - define the NN structure (# of input units, # of hidden units, etc.)
    - initialize the model’s parameters
    - loop
        - implement forward prop
        - compute loss
        - implement backward prop to get the gradients
        - update parameters (GD)
- in practice, helper function are build to compute steps 1-3, then merge them into one function called nn_model()
- once nn_model() is build and learned the right parameters, make predictions on the new data
- Defining the NN structure
    - Exercise 2 - layer_sizes
        - define three variables
            - n_x - the size oft he input layer
            - n_h - the size of the hidden layer (set this to 4, as n_h = 4, but only for this Exercise 2)
            - n_y - the size of the output layer
    - use shapes of X and Y to find n_x and n_y
    - hard code the hidden layer size to be 4

        ```python
        # GRADED FUNCTION: layer_sizes

        def layer_sizes(X, Y):
            """
            Arguments:
            X -- input dataset of shape (input size, number of examples)
            Y -- labels of shape (output size, number of examples)

            Returns:
            n_x -- the size of the input layer
            n_h -- the size of the hidden layer
            n_y -- the size of the output layer
            """
            #(≈ 3 lines of code)
            # n_x = ... 
            # n_h = ...
            # n_y = ... 
            # YOUR CODE STARTS HERE
            n_x = X.shape[0]
            n_h = 4
            n_y = Y.shape[0]
            # YOUR CODE ENDS HERE
            return (n_x, n_h, n_y)
        ```

- Initialize the model’s parameters
    - Exercise 3 - initialize_parameters
        - make sure the parameters’ sizes are right
        - initialize the weights matrices with random values
            - use np.random.randn(a,b) * 0.01 to randomly initialize a matrix of shape (a,b)
        - initialize the bias vectors as zeros
            - use np.zeros((a,b)) to initialize a matrix of shape (a,b) with zeros

        ```python
        # GRADED FUNCTION: initialize_parameters

        def initialize_parameters(n_x, n_h, n_y):
            """
            Argument:
            n_x -- size of the input layer
            n_h -- size of the hidden layer
            n_y -- size of the output layer

            Returns:
            params -- python dictionary containing your parameters:
                            W1 -- weight matrix of shape (n_h, n_x)
                            b1 -- bias vector of shape (n_h, 1)
                            W2 -- weight matrix of shape (n_y, n_h)
                            b2 -- bias vector of shape (n_y, 1)
            """    
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

- The loop
    - Exercise 4 - forward_propagation
        - implement forward_propagation()

        - check the mathematical representation of the classifier
        - use the function sigmoid()
        - use the function np.tanh()
        - implement
            - retrieve each parameter from the dictionary ‘parameters’ (which is the output of initialize_parameters() by using parameters[””])
            - implement forward prop, compute Z[1], A[1], Z[2], A[2] (the vector of all the prediction on all the examples in the training set)
            - values needed in the backpropagation are storedin ‘cache’
            - the cache will be given as an input to the backprop function

        ```python
        # GRADED FUNCTION:forward_propagation

        def forward_propagation(X, parameters):
            """
            Argument:
            X -- input data of size (n_x, m)
            parameters -- python dictionary containing your parameters (output of initialization function)

            Returns:
            A2 -- The sigmoid output of the second activation
            cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
            """
            # Retrieve each parameter from the dictionary "parameters"
            #(≈ 4 lines of code)
            # W1 = ...
            # b1 = ...
            # W2 = ...
            # b2 = ...
            # YOUR CODE STARTS HERE
            W1 = parameters["W1"]
            b1 = parameters["b1"] 
            W2 = parameters["W2"] 
            b2 = parameters["b2"] 
            # YOUR CODE ENDS HERE

            # Implement Forward Propagation to calculate A2 (probabilities)
            # (≈ 4 lines of code)
            # Z1 = ...
            # A1 = ...
            # Z2 = ...
            # A2 = ...
            # YOUR CODE STARTS HERE
            Z1 = W1@X + b1 # W1 = (n_h, n_x) -> (4, 2), X = (n_x, m) -> (2, 400)
            A1 = np.tanh(Z1)
            Z2 = W2@A1 + b2
            A2 = sigmoid(Z2)
            # YOUR CODE ENDS HERE

            assert(A2.shape == (1, X.shape[1]))

            cache = {"Z1": Z1,
                     "A1": A1,
                     "Z2": Z2,
                     "A2": A2}

            return A2, cache
        ```

- Compute the Cost
    - now the A[2] is computed, which contains a[2](i) for all examples, compute the cost function

    - Exercise 5 - compute_cost
        - Implement compute_cost() to compute the value of the cost J
        - there are many ways to implement the cross-entropy loss
        - this is one way to implement one part of the equation without for loops

            - logprobs = np.multiply(np.log(A2), Y)
            - cost = - np.sum(logprobs)
        - use either np.multiply() and then np.sum() or directly np.dot()
        - using np.multiply followed by the np.sum, the end result will be a type float, whereas with np.dot, the result will be a 2D numpy array
        - use  np.squeeze() to remove redundant dimensions (in case of single float, this will be reduced to a zero-dimension array
        - cast the array as a type float using float()

            ```python
            # GRADED FUNCTION: compute_cost

            def compute_cost(A2, Y):
                """
                Computes the cross-entropy cost given in equation (13)

                Arguments:
                A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
                Y -- "true" labels vector of shape (1, number of examples)

                Returns:
                cost -- cross-entropy cost given equation (13)

                """

                m = Y.shape[1] # number of examples

                # Compute the cross-entropy cost
                # (≈ 2 lines of code)
                # logprobs = ...
                # cost = ...
                # YOUR CODE STARTS HERE
                logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y),np.log(1-A2))
                cost = -1/m * np.sum(logprobs)
                # YOUR CODE ENDS HERE

                cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                                # E.g., turns [[17]] into 17 

                return cost
            ```

- Implement Backpropagation
    - using the cache computed during forward propagation, implement back prop
    - Exercise 6 - backward_propagation
    - backpropagation is usually the hardest (mathematically) part in deep learning

    - to compute dZ1, compute g[1]’(Z[1])
    - since g[1]() is the tanh activation function, if a = g[1](z) then g[1]’(z) = 1 - a@
    - compute g[1]’(Z[1]) using (1-np.power(A1, 2))

        ```python
        # GRADED FUNCTION: backward_propagation

        def backward_propagation(parameters, cache, X, Y):
            """
            Implement the backward propagation using the instructions above.

            Arguments:
            parameters -- python dictionary containing our parameters 
            cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
            X -- input data of shape (2, number of examples)
            Y -- "true" labels vector of shape (1, number of examples)

            Returns:
            grads -- python dictionary containing your gradients with respect to different parameters
            """
            m = X.shape[1]

            # First, retrieve W1 and W2 from the dictionary "parameters".
            #(≈ 2 lines of code)
            # W1 = ...
            # W2 = ...
            # YOUR CODE STARTS HERE
            W1 = parameters["W1"]
            W2 = parameters["W2"]    
            # YOUR CODE ENDS HERE

            # Retrieve also A1 and A2 from dictionary "cache".
            #(≈ 2 lines of code)
            # A1 = ...
            # A2 = ...
            # YOUR CODE STARTS HERE
            A1 = cache["A1"]
            A2 = cache["A2"]    
            # YOUR CODE ENDS HERE

            # Backward propagation: calculate dW1, db1, dW2, db2. 
            #(≈ 6 lines of code, corresponding to 6 equations on slide above)
            # dZ2 = ...
            # dW2 = ...
            # db2 = ...
            # dZ1 = ...
            # dW1 = ...
            # db1 = ...
            # YOUR CODE STARTS HERE
            dZ2 = A2 - Y
            dW2 = 1/m * np.dot(dZ2, A1.T)
            db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
            dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
            dW1 = 1/m * np.dot(dZ1, X.T)
            db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
            # YOUR CODE ENDS HERE

            grads = {"dW1": dW1,
                     "db1": db1,
                     "dW2": dW2,
                     "db2": db2}

            return grads
        ```

- Update Parameters
    - Exercise 7 - update_parameters
        - implement the update rule
        - use GD
        - use dW1, db1, dW2, db2 in order to update W1, b1, W2, b2
        - general gradient descent rule: theta = theta - alpha(dJ/dtheta), where alpha is the learning rate and theta represents a parameter

        - use copy.deepcopy(…) when copying lists or dictionaries that are passed as parameters to functions
        - it avoids input parameters being modified within the function

        ```python
        # GRADED FUNCTION: update_parameters

        def update_parameters(parameters, grads, learning_rate = 1.2):
            """
            Updates parameters using the gradient descent update rule given above

            Arguments:
            parameters -- python dictionary containing your parameters 
            grads -- python dictionary containing your gradients 

            Returns:
            parameters -- python dictionary containing your updated parameters 
            """
            # Retrieve a copy of each parameter from the dictionary "parameters". Use copy.deepcopy(...) for W1 and W2
            #(≈ 4 lines of code)
            # W1 = ...
            # b1 = ...
            # W2 = ...
            # b2 = ...
            # YOUR CODE STARTS HERE
            W1 = copy.deepcopy(parameters["W1"])
            b1 = parameters["b1"]
            W2 = copy.deepcopy(parameters["W2"])    
            b2 = parameters["b2"]
            # YOUR CODE ENDS HERE

            # Retrieve each gradient from the dictionary "grads"
            #(≈ 4 lines of code)
            # dW1 = ...
            # db1 = ...
            # dW2 = ...
            # db2 = ...
            # YOUR CODE STARTS HERE
            dW1 = grads["dW1"]
            db1 = grads["db1"]
            dW2 = grads["dW2"]
            db2 = grads["db2"]
            # YOUR CODE ENDS HERE

            # Update rule for each parameter
            #(≈ 4 lines of code)
            # W1 = ...
            # b1 = ...
            # W2 = ...
            # b2 = ...
            # YOUR CODE STARTS HERE
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2
            # YOUR CODE ENDS HERE

            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}

            return parameters
        ```

- Integration
    - integrate functions in nn_model()
    - Exercise 8 - nn_model
        - build the NN model in nn_model()
        - the NN model has to use the previous functions in the right order

        ```python
        # GRADED FUNCTION: nn_model

        def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
            """
            Arguments:
            X -- dataset of shape (2, number of examples)
            Y -- labels of shape (1, number of examples)
            n_h -- size of the hidden layer
            num_iterations -- Number of iterations in gradient descent loop
            print_cost -- if True, print the cost every 1000 iterations

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

            np.random.seed(3)
            n_x = layer_sizes(X, Y)[0]
            n_y = layer_sizes(X, Y)[2]

            # Initialize parameters
            #(≈ 1 line of code)
            # parameters = ...
            # YOUR CODE STARTS HERE
            parameters = initialize_parameters(n_x, n_h, n_y)
            # YOUR CODE ENDS HERE

            # Loop (gradient descent)

            for i in range(0, num_iterations):

                #(≈ 4 lines of code)
                # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
                # A2, cache = ...

                # Cost function. Inputs: "A2, Y". Outputs: "cost".
                # cost = ...

                # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
                # grads = ...

                # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
                # parameters = ...

                # YOUR CODE STARTS HERE
                A2, cache = forward_propagation(X, parameters)
                grads = backward_propagation(parameters, cache, X, Y)
                cost = compute_cost(A2, Y)
                parameters = update_parameters(parameters, grads)
                # YOUR CODE ENDS HERE

                # Print the cost every 1000 iterations
                if print_cost and i % 1000 == 0:
                    print ("Cost after iteration %i: %f" %(i, cost))

            return parameters

        '''
        Cost after iteration 0: 0.693497
        Cost after iteration 1000: 0.000180
        Cost after iteration 2000: 0.000088
        Cost after iteration 3000: 0.000058
        Cost after iteration 4000: 0.000044
        Cost after iteration 5000: 0.000035
        Cost after iteration 6000: 0.000029
        Cost after iteration 7000: 0.000025
        Cost after iteration 8000: 0.000022
        Cost after iteration 9000: 0.000019
        W1 = [[-0.48394893 -0.91443482]
         [-0.69227123 -1.28596651]
         [ 0.63806018  1.18124948]
         [ 0.73594679  1.35718308]
         [-0.62621054 -1.16022636]]
        b1 = [[ 0.00285386]
         [ 0.01642488]
         [-0.01114986]
         [-0.01656405]
         [ 0.01042274]]
        W2 = [[-1.45325879 -2.65768451  2.26286606  2.97274518 -2.18860534]]
        b2 = [[0.00758617]]
        '''
        ```

Test the Model

- Predict
    - Exercise 9 - predict
        - predict with the model by building predict()
        - use forward prop to predict results

        - to set the entries of a matrix X to 0 and 1 based on a threshold → X-new = (X > threshold)

        ```python
        # GRADED FUNCTION: predict

        def predict(parameters, X):
            """
            Using the learned parameters, predicts a class for each example in X

            Arguments:
            parameters -- python dictionary containing your parameters 
            X -- input data of size (n_x, m)

            Returns
            predictions -- vector of predictions of our model (red: 0 / blue: 1)
            """

            # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
            #(≈ 2 lines of code)
            # A2, cache = ...
            # predictions = ...
            # YOUR CODE STARTS HERE
            A2, cache = forward_propagation(X, parameters)
            predictions = (A2 > 0.5)
            # YOUR CODE ENDS HERE

            return predictions

         # Predictions: [[ True False  True]]
        ```

- Test the Model on the Planar Dataset
    - run the model and see how it performs on the planar dataset

        ```python
        # Build a model with a n_h-dimensional hidden layer
        parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

        # Plot the decision boundary
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        plt.title("Decision Boundary for hidden layer size " + str(4))

        '''
        Cost after iteration 0: 0.693162
        Cost after iteration 1000: 0.258625
        Cost after iteration 2000: 0.239334
        Cost after iteration 3000: 0.230802
        Cost after iteration 4000: 0.225528
        Cost after iteration 5000: 0.221845
        Cost after iteration 6000: 0.219094
        Cost after iteration 7000: 0.220661
        Cost after iteration 8000: 0.219409
        Cost after iteration 9000: 0.218485
        '''
        ```

        ```python
        # Print accuracy
        predictions = predict(parameters, X)
        print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

        # Accuracy: 90%
        ```

    - accuracy is really high compared to logistic regresion
    - the model has learned the patterns of the flower’s petals
    - NN are able to learn even highly non-linear decision boundaries

    Tuning hidden layer size

    - observe different behaviors of the model for various hidden layer sizes

        ```python
        # This may take about 2 minutes to run

        plt.figure(figsize=(16, 32))
        hidden_layer_sizes = [1, 2, 3, 4, 5]

        # you can try with different hidden layer sizes
        # but make sure before you submit the assignment it is set as "hidden_layer_sizes = [1, 2, 3, 4, 5]"
        # hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

        for i, n_h in enumerate(hidden_layer_sizes):
            plt.subplot(5, 2, i+1)
            plt.title('Hidden Layer of size %d' % n_h)
            parameters = nn_model(X, Y, n_h, num_iterations = 5000)
            plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
            predictions = predict(parameters, X)
            accuracy = float((np.dot(Y,predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size)*100)
            print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

        '''
        Accuracy for 1 hidden units: 67.5 %
        Accuracy for 2 hidden units: 67.25 %
        Accuracy for 3 hidden units: 90.75 %
        Accuracy for 4 hidden units: 90.5 %
        Accuracy for 5 hidden units: 91.25 %
        '''
        ```

    - the larger models are able to fit the training set better, until eventually the largest models overfit the data
    - the best hidden layer size seems to be around n_h = 5
    - indeed, a value around here seems to fit the data well without also incurring noticeable overfitting
    - regularization will allow larger models without overfitting

    Performance on other datasets

    ```python
    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    ### START CODE HERE ### (choose your dataset)
    dataset = "noisy_moons"
    ### END CODE HERE ###

    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        Y = Y%2

    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    ```