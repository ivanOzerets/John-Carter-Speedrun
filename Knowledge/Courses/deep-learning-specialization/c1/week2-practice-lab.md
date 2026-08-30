Packages

- numpy → fundamental package for scientific computing with Python
- h5py → common package to interact with a dataset that is stored on an H5 file
- matplotlib → famous library to plot graphs in Python
- PIL and scipy are used to test the model with a custom image

    ```python
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    import h5py
    import scipy
    from PIL import Image
    from scipy import ndimage
    from lr_utils import load_dataset
    from public_tests import *

    %matplotlib inline
    %load_ext autoreload
    %autoreload 2
    ```

Overview of the Problem set

- Problem statement: given a dataset ‘data.h5’ containing:
    - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
    - a test set of m_test images labeled as cat or non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels
    - each image is square (height = num_px) and (width = num_px)
- build a simple image-recognition alg that can correctly classify pictures as cat or non-cat
- load the data by running the following code

    ```python
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    ```

- add “_orig” at the end of image datasets (train and test) because we are going to preprocess them
- after preprocessing, end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don’t need any preprocessing)
- each line of train_set_x_orig and test_set_x_orig is an array representing an image
- visualize an example

    ```python
    # Example of a picture
    index = 25
    plt.imshow(train_set_x_orig[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    ```

- many software bugs in deep learning come from having matrix/vector dimensions that don’t fit
- keep matrix/vector dimensions straight
- Exercise 1
    - find the values for
        - m_train (number of training examples)
        - m_test (number of test examples)
        - num_px (height = width of a training image)
    - train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3)
    - access m_train by writing train_set_x_orig.shape[0]

    ```python
    #(≈ 3 lines of code)
    # m_train = 
    # m_test = 
    # num_px = 
    # YOUR CODE STARTS HERE
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_y.shape[1]
    num_px = train_set_x_orig.shape[1]
    # YOUR CODE ENDS HERE

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    ```

    - for convenience, reshape images of shape (num_px, num_px, 3) in a munpy-array of shape (num_px * num_px * 3, 1)
    - after this, the training dataset is a numpy-array where each column represents a flattened image
    - there should be m_train columns
- Exercise 2
    - reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into a single vector
    - a trick for flattening a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d, a) is to use: X_flatten = X.reshape(X.shape[0], -1).T
    - so shape gives you the product of the other dimensions

    ```python
    # Reshape the training and test examples
    #(≈ 2 lines of code)
    # train_set_x_flatten = ...
    # test_set_x_flatten = ...
    # YOUR CODE STARTS HERE
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    print(train_set_x_flatten.shape, test_set_x_flatten.shape)
    # YOUR CODE ENDS HERE
    # i think train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T istead of 
    #         train_set_x_orig.reshape(-1, train_set_x_orig.shape[0]) because
    # you're trying to keep the first dimension

    # Check that the first 10 pixels of the second image are in the correct place
    assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
    assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

    '''
    (12288, 209) (12288, 50)
    train_set_x_flatten shape: (12288, 209)
    train_set_y shape: (1, 209)
    test_set_x_flatten shape: (12288, 50)
    test_set_y shape: (1, 50)
    '''
    ```

    - to represent color images, the red, green and blue channels must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255
    - one common preprocessing step in ML is to center and standardize the dataset, meaning to substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array
    - for picture dataset, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel

    ```python
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.
    ```

    - steps to remember in pre-processing data
        - figure out the dimensions and shapes of the problem
        - reshape the datasets such that each example is now a vector
        - standardize the data

General Architecture of the learning alg

- design a simple alg to distinguish cat images from non-cat iamges
- build a logistic regression, using a NN mindset

- Mathematical expression of the alg
    - for one example

    - the cost is then computed by summing over all training examples

- Key steps
    - initialize the parameters of the model
    - learn the parameters for the model by minimizing the cost
    - use the learned parameters to make predications (on the test set)
    - analyze the results and conclude

Building the parts of the algorithm

- the main steps for building a NN are
    - define the model structure (such as number of input features)
    - initialize the model’s parameters
    - loop
        - calculate current loss (forward propagation)
        - calculate current gradient (backward propagation)
        - update parameters (gradient descent)
- often 1-3 separate NN and integrate into a model()
- Helper function
    - Exercise 3 - sigmoid
        - implement sigmoid()

        ```python
        # GRADED FUNCTION: sigmoid

        def sigmoid(z):
            """
            Compute the sigmoid of z

            Arguments:
            z -- A scalar or numpy array of any size.

            Return:
            s -- sigmoid(z)
            """

            #(≈ 1 line of code)
            # s = ...
            # YOUR CODE STARTS HERE
            s = 1/(1+np.exp(-z))    
            # YOUR CODE ENDS HERE

            return s
        ```

- Initializing parameters
    - Exercise 4 - initialize_with_zeros
        - implement parameter initialization
        - initialize w as a vector of zeros

        ```python
        # GRADED FUNCTION: initialize_with_zeros

        def initialize_with_zeros(dim):
            """
            This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

            Argument:
            dim -- size of the w vector we want (or number of parameters in this case)

            Returns:
            w -- initialized vector of shape (dim, 1)
            b -- initialized scalar (corresponds to the bias) of type float
            """

            # (≈ 2 lines of code)
            # w = ...
            # b = ...
            # YOUR CODE STARTS HERE
            w = np.zeros((dim, 1))
            b = 0.
            # YOUR CODE ENDS HERE

            return w, b
        ```

- Forward and Backward propagation
    - perform the forward and backward propagation steps for learning the parameters
    - Exercise 5 - propagate
        - implement a function propagate() that computes the cost function and its gradient
        - forward propagation
            - get X
            - compute A = sigmoid(wTX + b) = (a1, a2, …, a(m-1), a(m))
            - calculate the cost function

        ```python
        # GRADED FUNCTION: propagate

        def propagate(w, b, X, Y):
            """
            Implement the cost function and its gradient for the propagation explained above

            Arguments:
            w -- weights, a numpy array of size (num_px * num_px * 3, 1)
            b -- bias, a scalar
            X -- data of size (num_px * num_px * 3, number of examples)
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

            Return:
            grads -- dictionary containing the gradients of the weights and bias
                    (dw -- gradient of the loss with respect to w, thus same shape as w)
                    (db -- gradient of the loss with respect to b, thus same shape as b)
            cost -- negative log-likelihood cost for logistic regression

            Tips:
            - Write your code step by step for the propagation. np.log(), np.dot()
            """

            m = X.shape[1]

            # FORWARD PROPAGATION (FROM X TO COST)
            #(≈ 2 lines of code)
            # compute activation
            # A = ...
            # compute cost by using np.dot to perform multiplication. 
            # And don't use loops for the sum.
            # cost = ...                                
            # YOUR CODE STARTS HERE
            A = sigmoid(np.dot(w.T, X) + b)
            cost = -(1/m) * np.sum((Y*np.log(A) + (1 - Y)*np.log(1-A)))   
            # YOUR CODE ENDS HERE

            # BACKWARD PROPAGATION (TO FIND GRAD)
            #(≈ 2 lines of code)
            # dw = ...
            # db = ...
            # YOUR CODE STARTS HERE
            dw = 1/m * np.dot(X,(A-Y).T)
            db = 1/m * np.sum(A-Y)
            # YOUR CODE ENDS HERE
            cost = np.squeeze(np.array(cost))

            grads = {"dw": dw,
                     "db": db}

            return grads, cost
        ```

    - a little confused as to when matrix multiplication or dot
- Optimization
    - Exercise 6 - optimize
        - write down the optimization function
        - learn w and b by minimizing the cost function J
        - for a parameter theta, the update rule is theta = theta - alpha * dtheta, where alpha is the learning rate

        ```python
        # GRADED FUNCTION: optimize

        def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
            """
            This function optimizes w and b by running a gradient descent algorithm

            Arguments:
            w -- weights, a numpy array of size (num_px * num_px * 3, 1)
            b -- bias, a scalar
            X -- data of shape (num_px * num_px * 3, number of examples)
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
            num_iterations -- number of iterations of the optimization loop
            learning_rate -- learning rate of the gradient descent update rule
            print_cost -- True to print the loss every 100 steps

            Returns:
            params -- dictionary containing the weights w and bias b
            grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
            costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

            Tips:
            You basically need to write down two steps and iterate through them:
                1) Calculate the cost and the gradient for the current parameters. Use propagate().
                2) Update the parameters using gradient descent rule for w and b.
            """

            w = copy.deepcopy(w)
            b = copy.deepcopy(b)

            costs = []

            for i in range(num_iterations):
                # (≈ 1 lines of code)
                # Cost and gradient calculation 
                # grads, cost = ...
                # YOUR CODE STARTS HERE
                grads, cost = propagate(w, b, X, Y)        
                # YOUR CODE ENDS HERE

                # Retrieve derivatives from grads
                dw = grads["dw"]
                db = grads["db"]

                # update rule (≈ 2 lines of code)
                # w = ...
                # b = ...
                # YOUR CODE STARTS HERE
                w = w - learning_rate * dw
                b = b - learning_rate * db
                # YOUR CODE ENDS HERE

                # Record the costs
                if i % 100 == 0:
                    costs.append(cost)

                    # Print the cost every 100 training iterations
                    if print_cost:
                        print ("Cost after iteration %i: %f" %(i, cost))

            params = {"w": w,
                      "b": b}

            grads = {"dw": dw,
                     "db": db}

            return params, grads, costs
        ```

Exercise 7 - predict

- the previous function will output the learned w and b
- w and b are used to predict the labels for a dataset X
- implement the predict() function
    - calculate Y_hat = A = sigmoid(wT * X + b)
    - convert the entries of a into 0 (if activation ≤ 0.5) or 1 (if activation > 0.5), stores the predications in a vector Y_prediction

    ```python
    # GRADED FUNCTION: predict

    def predict(w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        #(≈ 1 line of code)
        # A = ...
        # YOUR CODE STARTS HERE
        A = sigmoid(np.dot(w.T, X) + b)    
        # YOUR CODE ENDS HERE

        for i in range(A.shape[1]):

            # Convert probabilities A[0,i] to actual predictions p[0,i]
            #(≈ 4 lines of code)
            # if A[0, i] > ____ :
            #     Y_prediction[0,i] = 
            # else:
            #     Y_prediction[0,i] = 
            # YOUR CODE STARTS HERE
            break

        Y_prediction = (A > 0.5).astype(int)
            # YOUR CODE ENDS HERE

        return Y_prediction
    ```

Merge all functions into a model

- put it all together
- Exercise 8 - model
    - implement the model function
    - Y_predication_test for the predictions on the test set
    - Y_prediction_train for the prediction on the train set
    - parameters, grads, costs for the outputs of optimize()

    ```python
    # GRADED FUNCTION: model

    def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to True to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """
        # (≈ 1 line of code)   
        # initialize parameters with zeros
        # and use the "shape" function to get the first dimension of X_train
        # w, b = ...

        #(≈ 1 line of code)
        # Gradient descent 
        # params, grads, costs = ...

        # Retrieve parameters w and b from dictionary "params"
        # w = ...
        # b = ...

        # Predict test/train set examples (≈ 2 lines of code)
        # Y_prediction_test = ...
        # Y_prediction_train = ...

        # YOUR CODE STARTS HERE
        w, b = np.zeros((X_train.shape[0], 1)), 0.

        params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

        w = params["w"]
        b = params["b"]

        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)
        # YOUR CODE ENDS HERE

        # Print train/test Errors
        if print_cost:
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}

        return d
    ```

- training accuracy is close to 100%
- this is a good sanity check: the model is working and has high enough capacity to fit the training data
- test accuracy is 7-%
- it is actually not bad for this simple model, given the small dataset used and that logicstic regression is a linear classifier
- the model is clearly overfitting the training data
- regularization can be used to reduce overfitting

    ```python
    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

    '''
    Cost after iteration 0: 0.693147
    Cost after iteration 100: 0.584508
    ...
    Cost after iteration 1800: 0.146542
    Cost after iteration 1900: 0.140872
    train accuracy: 99.04306220095694 %
    test accuracy: 70.0 %

    '''
    ```

    ```python
    # Example of a picture that was wrongly classified.
    index = 1
    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")
    ```

- plot the cost function and the gradients

    ```python
    # Plot learning curve (with costs)
    costs = np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    plt.show()
    ```

- interpretation: the cost decreasing shows that the parameters are being learned
- however, the model could be trained on the training set even more
- if you run the model for longer, the training set accuracy could go up, but the tests et accuracy goes down → overfitting

Further analysis

- Choice of learning rate
    - in order for GD to work, one must choose the learning rate wisely
    - the learning rate alpha determines how rapidly the parameters update
    - if the learning rate is too large, the optimal value will be ‘overshot’
    - if the learning rate is too small, the convergence will be too slow
    - well-tuned learning rate is crucial
    - compare learning curve of the model with other choices

    ```python
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}

    for lr in learning_rates:
        print ("Training a model with learning rate: " + str(lr))
        models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for lr in learning_rates:
        plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    ```

- interpretation
    - different learnign rates give different costs and thus different predictions results
    - if the learning rate is too large (0.01), the cost may oscillate up and down
    - it may even diverge
    - a lower cost doesn’t mean a better model
    - check if there is possibly overfitting
    - overfitting happens when the training accuracy is a lot higher than the test accuracy
    - in DL, it is recommended to
        - choose the learning rate that bettter minimized th cost rfunction’
        - if the model overfits, use other techniques to reduce overfitting

Test with custom image

- test own cat image

    ```python
    # change this to the name of your image file
    my_image = "my_image.jpg"   

    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    ```