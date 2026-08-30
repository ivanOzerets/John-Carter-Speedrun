Packages

- numpy
- matplotlib
- h5py is a common package to interact with a dataset that is stored on an H5 file
- PIL and scipy are used to test the model with a custom picture

    ```python
    import time
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    import scipy
    from PIL import Image
    from scipy import ndimage
    from dnn_app_utils_v3 import *
    from public_tests import *

    %matplotlib inline
    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    %load_ext autoreload
    %autoreload 2

    np.random.seed(1)
    ```

Load and Process the Dataset

- use the same “Cat vs non-Cat” dataset
- the model had previously 70% test accuracy on classifying cat vs non-cat images
- problem statement
    - a training set of ‘m_train’ images labelled as cat or non-cat
    - a test set of ‘m_test’ images albelled as cat and non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB)
- load the data

    ```python
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    ```

- show an image in the dataset

    ```python
    # Example of a picture
    index = 10
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    ```

    ```python
    # Explore your dataset 
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    '''
    Number of training examples: 209
    Number of testing examples: 50
    Each image is of size: (64, 64, 3)
    train_x_orig shape: (209, 64, 64, 3)
    train_y shape: (1, 209)
    test_x_orig shape: (50, 64, 64, 3)
    test_y shape: (1, 50)
    '''
    ```

- reshape and standardize the images before feeding them to the network

    ```python
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    '''
    train_x's shape: (12288, 209)
    test_x's shape: (12288, 50)
    '''
    ```

- 12,288 equals 64x64x3, which is the size of one reshaped image vector

Model Architecture

- 2-layer NN
    - build a NN to distinguish cat images from non-cat images
    - build two different models
        - a 2-layer NN
        - an L-layer deep NN
    - then, compare the performance of the models, and try out some different values for L

    - the input is a (64,64,3) image which is flattened to a vector of size (12288,1)
    - the corresponding vector: [x0, …, x12287].T is then multiplied by the weight matrix W[1] of size (n[1], 12288)
    - add a bias term and take its relu to get the following vector: [a[1]0, a[1]1, …, a[1]n[1]-1].T
    - multiply the resulting vector by W[2] and add the intercept (bias)
    - take the sigmoid of the result, if it’s greater than 0.5, classify it as a cat
- L-layer Deep NN
    - difficult to represent an L-layer deep NN using the above representation

    - the input is a (64,64,3) image which is flattened to a vector of size (12288,1)
    - the corresponding vector: [x0, x12287].T is then multiplied by the weight matrix W[1] and then add the intercept b[1]
    - the result is called the linear unit
    - take the relu of the linear unit, this process could be repeated several times for each (W[l], b[l]) depending on the model architecture
    - take the sigmoid of the final linear unit, if it is greater than 0.5, classify as a cat
- General Methodology
    - follow the deep learning methodology to build the model
        - initialize parameters / define hyperparameters
        - loop for num_iterations
            - forward propagation
            - compute cost function
            - backward propagation
            - update parameters
        - use trained parameters to predict labels

Two-layer NN

- Exercise 1 - two_layer_model
    - use the helper function from the assignment before to build a 2-layer NN with the following structure: LINEAR → RELU → LINEAR → SIGMOID

        ```python
        def initialize_parameters(n_x, n_h, n_y):
            ...
            return parameters 
        def linear_activation_forward(A_prev, W, b, activation):
            ...
            return A, cache
        def compute_cost(AL, Y):
            ...
            return cost
        def linear_activation_backward(dA, cache, activation):
            ...
            return dA_prev, dW, db
        def update_parameters(parameters, grads, learning_rate):
            ...
            return parameters
        ```

        ```python
        ### CONSTANTS DEFINING THE MODEL ####
        n_x = 12288     # num_px * num_px * 3
        n_h = 7
        n_y = 1
        layers_dims = (n_x, n_h, n_y)
        learning_rate = 0.0075
        ```

        ```python
        # GRADED FUNCTION: two_layer_model

        def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
            """
            Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

            Arguments:
            X -- input data, of shape (n_x, number of examples)
            Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
            layers_dims -- dimensions of the layers (n_x, n_h, n_y)
            num_iterations -- number of iterations of the optimization loop
            learning_rate -- learning rate of the gradient descent update rule
            print_cost -- If set to True, this will print the cost every 100 iterations 

            Returns:
            parameters -- a dictionary containing W1, W2, b1, and b2
            """

            np.random.seed(1)
            grads = {}
            costs = []                              # to keep track of the cost
            m = X.shape[1]                           # number of examples
            (n_x, n_h, n_y) = layers_dims

            # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
            #(≈ 1 line of code)
            # parameters = ...
            # YOUR CODE STARTS HERE
            parameters = initialize_parameters(n_x, n_h, n_y)    
            # YOUR CODE ENDS HERE

            # Get W1, b1, W2 and b2 from the dictionary parameters.
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]

            # Loop (gradient descent)

            for i in range(0, num_iterations):

                # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
                #(≈ 2 lines of code)
                # A1, cache1 = ...
                # A2, cache2 = ...
                # YOUR CODE STARTS HERE
                A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
                A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
                # YOUR CODE ENDS HERE

                # Compute cost
                #(≈ 1 line of code)
                # cost = ...
                # YOUR CODE STARTS HERE
                cost = compute_cost(A2, Y)        
                # YOUR CODE ENDS HERE

                # Initializing backward propagation
                dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

                # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
                #(≈ 2 lines of code)
                # dA1, dW2, db2 = ...
                # dA0, dW1, db1 = ...
                # YOUR CODE STARTS HERE
                dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
                dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
                # YOUR CODE ENDS HERE

                # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
                grads['dW1'] = dW1
                grads['db1'] = db1
                grads['dW2'] = dW2
                grads['db2'] = db2

                # Update parameters.
                #(approx. 1 line of code)
                # parameters = ...
                # YOUR CODE STARTS HERE
                parameters = update_parameters(parameters, grads, learning_rate)        
                # YOUR CODE ENDS HERE

                # Retrieve W1, b1, W2, b2 from parameters
                W1 = parameters["W1"]
                b1 = parameters["b1"]
                W2 = parameters["W2"]
                b2 = parameters["b2"]

                # Print the cost every 100 iterations and for the last iteration
                if print_cost and (i % 100 == 0 or i == num_iterations - 1):
                    print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
                if i % 100 == 0:
                    costs.append(cost)

            return parameters, costs

        def plot_costs(costs, learning_rate=0.0075):
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        ```

- Train the model
    - the cost should decrease on every iteration
    - it may take up to 5 min to run 2500 iterations

    ```python
    parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    plot_costs(costs, learning_rate)

    '''
    Cost after iteration 0: 0.693049735659989
    Cost after iteration 100: 0.6464320953428849
    Cost after iteration 200: 0.6325140647912677
    ...
    Cost after iteration 2400: 0.04855478562877019
    Cost after iteration 2499: 0.04421498215868956
    '''
    ```

- if non-vectorized, it might have taken 10 times longer to train
- use the trained parameters to classify images from the dataset
- to see the prediction on the training and test sets

    ```python
    predictions_train = predict(train_x, train_y, parameters)

    # Accuracy: 0.9999999999999998
    ```

    ```python
    predictions_test = predict(test_x, test_y, parameters)

    # **Accuracy: 0.72**
    ```

- the 2-layer NN performed slightly better than the logistic regression implementation

L-layer NN

- Exercise 2 - L_layer_model
    - build an L-layer NN with the following structure [LINEAR -> RELU]×(L-1) -> LINEAR -> SIGMOID

        ```python
        def initialize_parameters_deep(layers_dims):
            ...
            return parameters 
        def L_model_forward(X, parameters):
            ...
            return AL, caches
        def compute_cost(AL, Y):
            ...
            return cost
        def L_model_backward(AL, Y, caches):
            ...
            return grads
        def update_parameters(parameters, grads, learning_rate):
            ...
            return parameters
        ```

        ```python
        ### CONSTANTS ###
        layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
        ```

        ```python
        # GRADED FUNCTION: L_layer_model

        def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
            """
            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

            Arguments:
            X -- input data, of shape (n_x, number of examples)
            Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

            np.random.seed(1)
            costs = []                         # keep track of cost

            # Parameters initialization.
            #(≈ 1 line of code)
            # parameters = ...
            # YOUR CODE STARTS HERE
            parameters = initialize_parameters_deep(layers_dims)    
            # YOUR CODE ENDS HERE

            # Loop (gradient descent)
            for i in range(0, num_iterations):

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                #(≈ 1 line of code)
                # AL, caches = ...
                # YOUR CODE STARTS HERE
                AL, caches = L_model_forward(X, parameters)        
                # YOUR CODE ENDS HERE

                # Compute cost.
                #(≈ 1 line of code)
                # cost = ...
                # YOUR CODE STARTS HERE
                cost = compute_cost(AL, Y)        
                # YOUR CODE ENDS HERE

                # Backward propagation.
                #(≈ 1 line of code)
                # grads = ...    
                # YOUR CODE STARTS HERE
                grads = L_model_backward(AL, Y, caches)        
                # YOUR CODE ENDS HERE

                # Update parameters.
                #(≈ 1 line of code)
                # parameters = ...
                # YOUR CODE STARTS HERE
                parameters = update_parameters(parameters, grads, learning_rate)        
                # YOUR CODE ENDS HERE

                # Print the cost every 100 iterations and for the last iteration
                if print_cost and (i % 100 == 0 or i == num_iterations - 1):
                    print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
                if i % 100 == 0:
                    costs.append(cost)

            return parameters, costs
        ```

- Train the model
    - train the model as a 4-layer NN
        - the cost should decrease on every iteration
        - may take up to 5 min to run

        ```python
        parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

        '''
        Cost after iteration 0: 0.7717493284237686
        Cost after iteration 100: 0.6720534400822914
        Cost after iteration 200: 0.6482632048575212
        ...
        Cost after iteration 2400: 0.09287821526472398
        Cost after iteration 2499: 0.08843994344170202
        '''
        ```

        ```python
        pred_train = predict(train_x, train_y, parameters)

        # Accuracy: 0.9856459330143539
        ```

        ```python
        pred_test = predict(test_x, test_y, parameters)

        # Accuracy: 0.8
        ```

Result Analysis

- some images the L-layer model labeled incorrectly

    ```python
    print_mislabeled_images(classes, test_x, test_y, pred_test)
    ```

- few types of images the model tends to do poorly on
    - cat body in an unusual position
    - cat appears against a background of a similar color
    - unusual cat color and species
    - camera angle
    - brightness of the picture
    - scale variation

Test with your own image

```python
## START CODE HERE ##
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T

my_predicted_image = predict(image, my_label_y, parameters)

print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```