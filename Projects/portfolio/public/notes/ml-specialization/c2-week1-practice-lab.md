Packages

- run cells below to import all the packages needed
    - numpy for scientific computing with Python
    - matplotlib to plot graphs in Python
    - tensorflow for machine learning

    ```python
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    import matplotlib.pyplot as plt
    from autils import *
    %matplotlib inline

    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
    ```

- Tensorflow and Keras
    - Tensorflow is a machine learning package developed by Google
    - in 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0
    - Keras is a framework developed independently by Francois Chollet that creates a simple, layer-centric interface to Tensorflow

Neural Networks

- extended to handle non-linear boundaries using polynomial regression
- for even more complex scenarios such as image recognition, neural networks are preferred
- Problem Statement
    - use a NN to recognize two handwritten digits, zero and one
    - this is a binary classification task
    - automated handwritten digit recognition is widely used today from recognizing zip codes on mail envelopes to recognizing amounts written on bank checks
    - extend this network to recognize all 10 digits in a future assignment
- Dataset
    - load the dataset
    - the load_data() function loads the data into variables X and y
    - the data set contains 1000 training examples of handwritten digits, here limited to zero and one
        - each training example is a 20 pixel x 20 pixel grayscale image of the digit
            - each pixel is represented by a floating-point number indicating the grayscale intensity at that location
            - the 20 by 20 grid of pixels is “unrolled” into a 400D vector
            - each training example becomes a single row in our data matrix X
            - this gives us a 1000 x 400 matrix X where every row is a training example of a handwritten digit image

    - the second part of the training set is a 1000 x 1 D vector y that contains labels for the training set
        - y = 0 if the image is a of the digit 0, y = 1 if the image is of the digit 1
    - this is a subset of the MNIST handwritten digit dataset ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))

        ```python
        http://yann.lecun.com/exdb/mnist/
        ```

    - View the variables
        - a good place to start is to print out each variable and see what it contains

        ```python
        print ('The first element of X is: ', X[0])

        '''
        A bunch of numbers and zeros like:
        1.94035948e-06 -7.37438725e-04 -8.13403799e-03 -1.86104473e-02
        '''
        ```

        ```python
        print ('The first element of y is: ', y[0,0])
        print ('The last element of y is: ', y[-1,0])

        '''
        The first element of y is:  0
        The last element of y is:  1
        '''
        ```

    - Check the dimensions of the variables
        - view the dimensions
        - print the shape of X and y and see how many training examples there are

        ```python
        print ('The shape of X is: ' + str(X.shape))
        print ('The shape of y is: ' + str(y.shape))

        '''
        The shape of X is: (1000, 400)
        The shape of y is: (1000, 1)
        '''
        ```

    - Visualizing the data
        - begin by visualizing a subset of the training set
        - randomly selects 64 rows from X, map each row back to a 20 pixel x 20 pixel grayscale image and display the images together

        ```python
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # You do not need to modify anything in this cell

        m, n = X.shape

        fig, axes = plt.subplots(8,8, figsize=(8,8))
        fig.tight_layout(pad=0.1)

        for i,ax in enumerate(axes.flat):
            # Select random indices
            random_index = np.random.randint(m)

            # Select rows corresponding to the random indices and
            # reshape the image
            X_random_reshaped = X[random_index].reshape((20,20)).T # interesting how the reshaped image is then tranformed?

            # Display the image
            ax.imshow(X_random_reshaped, cmap='gray')

            # Display the label above the image
            ax.set_title(y[random_index,0])
            ax.set_axis_off()
        ```

- Model representation
    - the NN is in the figure below
    - three dense layers with sigmoid activations
    - inputs are pixel values of digit images
    - the images are of size 20 x 20, this gives 400 inputs

    - the parameters have dimensions that are sized for a NN with 25 units in layer 1, 15 units in layer 2 and 1 output unit in layer 3
        - the dimensions of these parameters are determined as follows:
        if network has s_in units in a layer and s_out units in the next layer, then
            - W will be of dimension s_in x s_out
            - b will be a vector with s_out elements
        - the shapes of W and b are
            - layer1: shape of W1 is (400, 25), and shape of b1 is (25,)
            - layer2: shape of W2 is (25, 15) and shape of b2 is (15,)
            - layer3: shape of W3 is (15,1) and shape of b3 is (1,)
        - Note: the bias vector b could be represented as a 1D (n,) or 2D (1,n) array, Tensorfow utilized a 1D representation

Tensorflow Model Implementation

- Tensorflow models are built layer by layer
- specify a layer’s output dimensions and this determines the next layer’s input dimension
- the input dimension of the first layer is derived from the size of the input data specified in the model.fit() statment below
- Note: it is also possible to add an input layer that specifies the input dimension of the first layer, for examples: tf.keras.Input(shape=(400,))

Exercise 1

- use Keras Sequential model and Dense Layer with a sigmoid activation

    ```python
    # UNQ_C1
    # GRADED CELL: Sequential model

    model = Sequential(
        [               
            tf.keras.Input(shape=(400,)),    #specify input size
            ### START CODE HERE ### 
            Dense(25, activation='sigmoid'),
            Dense(15, activation='sigmoid'),
            Dense(1, activation='sigmoid')
            ### END CODE HERE ### 
        ], name = "my_model" 
    )                            

    ```

    ```python
    model.summary()

    '''
    Model: "my_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 25)                10025     

     dense_1 (Dense)             (None, 15)                390       

     dense_2 (Dense)             (None, 1)                 16        

    =================================================================
    Total params: 10,431
    Trainable params: 10,431
    Non-trainable params: 0
    _________________________________________________________________
    '''
    ```

- model.summary() function displays a useful summary of the model
- we have specified an input layer size, shape of the weight and bias array are determined and the total number of parameters per layer can be shown
- the parameters count shown in the summary correspond to the number of elements in teh weight and bias arrays

    ```python
    L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
    L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
    L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )

    # L1 params =  10025 , L2 params =  390 ,  L3 params =  16
    ```

- examine details of the model by first extracting the layers with model.layers and then extracting the weights with layerx.getweights()

    ```python
    [layer1, layer2, layer3] = model.layers
    ```

    ```python
    #### Examine Weights shapes
    W1,b1 = layer1.get_weights()
    W2,b2 = layer2.get_weights()
    W3,b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    '''
    W1 shape = (400, 25), b1 shape = (25,)
    W2 shape = (25, 15), b2 shape = (15,)
    W3 shape = (15, 1), b3 shape = (1,)
    '''
    ```

- xx.get_weights return a numpy array
- one can also access the weights directly in their tensor form
- note the shape of the tensors in the final layer

    ```python
    print(model.layers[2].weights)

    '''
    [<tf.Variable 'dense_5/kernel:0' shape=(15, 1) dtype=float32, numpy=
    array([[-0.3770929 ],
           [-0.51207316],
           [ 0.52808315],
           [-0.32897332],
           [-0.16887173],
           [ 0.33763015],
           [ 0.41125625],
           [ 0.24390078],
           [-0.48282987],
           [-0.37446314],
           [-0.47436196],
           [ 0.1536209 ],
           [-0.43464255],
           [-0.20207965],
           [ 0.45448714]], dtype=float32)>, <tf.Variable 'dense_5/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
    '''
    ```

- define a loss function and run GD to fit the weights of the model tot he training data

    ```python
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        X,y,
        epochs=20
    )

    '''
    Epoch 1/20
    32/32 [==============================] - 0s 1ms/step - loss: 0.6524
    Epoch 2/20
    32/32 [==============================] - 0s 2ms/step - loss: 0.4720
    ...
    Epoch 19/20
    32/32 [==============================] - 0s 1ms/step - loss: 0.0219
    Epoch 20/20
    32/32 [==============================] - 0s 2ms/step - loss: 0.0206
    <keras.callbacks.History at 0x765c6cd1cc50>
    '''
    ```

- to run the model on an example to make a prediction, use Keras predict
- the input to predict is an array so the single example is reshaped to be 2D

    ```python
    prediction = model.predict(X[0].reshape(1,400))  # a zero
    print(f" predicting a zero: {prediction}")
    prediction = model.predict(X[500].reshape(1,400))  # a one
    print(f" predicting a one:  {prediction}")

    # predicting a zero: [[0.01112515]]
    # predicting a one:  [[0.9812052]]
    ```

- the output of the model is interpreted as a probability
- the input is a zero
- the model predicts the probability that the input is a one is nearly zero
- the second example, the input is a one
- the model predicts the probability that the input is a one is nearly one
- as in the case of logistic regression, the probability is compared to a threshold to make a final prediction

    ```python
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print(f"prediction after threshold: {yhat}")

    #prediction after threshold: 1
    ```

- compare the predictions vs the labels for a random sample of 64 digits

    ```python
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,400))
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0

        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()
    ```

Numpy Model Implementation (Forward Prop in Numpy)

- it is possible to build a dense layer using numpy
- can then be utilized to build a multi-layer NN

Exercise 2

- build a dense layer subroutine
- example in lecture utilized a for loop to visit each unit j in the layer and perform the dot product of the weights for that unit W[:,j] and sum the bias for the unit b[j] to form z
- an activation function g(z) is then applied to that result
- does not utilize some of the matrix operation

    ```python
    # UNQ_C2
    # GRADED FUNCTION: my_dense

    def my_dense(a_in, W, b, g):
        """
        Computes dense layer
        Args:
          a_in (ndarray (n, )) : Data, 1 example 
          W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
          b    (ndarray (j, )) : bias vector, j units  
          g    activation function (e.g. sigmoid, relu..)
        Returns
          a_out (ndarray (j,))  : j units
        """
        units = W.shape[1]
        a_out = np.zeros(units)

        ### START CODE HERE ### 
        for j in range(units):
            w = W[:,j]
            z = np.dot(w, a_in) + b[j]
            a_out[j] = g(z)
        ### END CODE HERE ### 

        return(a_out)
    ```

    ```python
    # Quick Check
    x_tst = 0.1*np.arange(1,3,1).reshape(2,)  # (1 examples, 2 features)
    W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
    b_tst = 0.1*np.arange(1,4,1).reshape(3,)  # (3 features)
    A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)
    print(A_tst)

    # [0.54735762 0.57932425 0.61063923]
    ```

- builds a three-layer neural network utilizing the my_dense() subroutine

    ```python
    W1_tmp,b1_tmp = layer1.get_weights()
    W2_tmp,b2_tmp = layer2.get_weights()
    W3_tmp,b3_tmp = layer3.get_weights()
    ```

- copy trained weights and biases from Tensorflow

    ```python
    # make predictions
    prediction = my_sequential(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print( "yhat = ", yhat, " label= ", y[0,0])
    prediction = my_sequential(X[500], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print( "yhat = ", yhat, " label= ", y[500,0])

    # yhat =  0  label=  0
    # yhat =  1  label=  1
    ```

- predictions from both the numpy model and the tensorflow model

    ```python
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network implemented in Numpy
        my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
        my_yhat = int(my_prediction >= 0.5)

        # Predict using the Neural Network implemented in Tensorflow
        tf_prediction = model.predict(X[random_index].reshape(1,400))
        tf_yhat = int(tf_prediction >= 0.5)

        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{tf_yhat},{my_yhat}")
        ax.set_axis_off() 
    fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
    plt.show()
    ```

Vectorized Numpy Model Implementation

- below describes a layer operation that computes the output for all units in a layer on a given input example

- we can demonstrate using the examples X and the W1, b1 parameters
- use np.matmul to perform the matrix multiply
- the dimensions of x and W must be compatible as shown in the diagram

    ```python
    x = X[0].reshape(-1,1)         # column vector (400,1)
    z1 = np.matmul(x.T,W1) + b1    # (1,400)(400,25) = (1,25)
    a1 = sigmoid(z1)
    print(a1.shape)

    # (1, 25)
    ```

- compute all the units for all example in one matrix-matrix operation

- the full operation is Z = XW + b
- utilizes numpy broadcasting to expand b to m rows

Exercise 3

- compose a new my_dense_v subroutine that performs the layer calculations for a matrix of examples

    ```python
    # UNQ_C3
    # UNGRADED FUNCTION: my_dense_v

    def my_dense_v(A_in, W, b, g):
        """
        Computes dense layer
        Args:
          A_in (ndarray (m,n)) : Data, m examples, n features each
          W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
          b    (ndarray (1,j)) : bias vector, j units  
          g    activation function (e.g. sigmoid, relu..)
        Returns
          A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
        """

        ### START CODE HERE ###
        z = np.matmul(A_in, W) + b
        A_out = g(z)
        ### END CODE HERE ### 

        return(A_out)
    ```

    ```python
    X_tst = 0.1*np.arange(1,9,1).reshape(4,2) # (4 examples, 2 features)
    W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
    b_tst = 0.1*np.arange(1,4,1).reshape(1,3) # (1,3 features)
    A_tst = my_dense_v(X_tst, W_tst, b_tst, sigmoid)
    print(A_tst)

    '''
    [[0.54735762 0.57932425 0.61063923]
     [0.57199613 0.61301418 0.65248946]
     [0.5962827  0.64565631 0.6921095 ]
     [0.62010643 0.67699586 0.72908792]]
    '''
    ```

- the following cell builds a three-layer NN utilizing the my_dense_v subroutine

    ```python
    def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
        A1 = my_dense_v(X,  W1, b1, sigmoid)
        A2 = my_dense_v(A1, W2, b2, sigmoid)
        A3 = my_dense_v(A2, W3, b3, sigmoid)
        return(A3)
    ```

- copy trained weight and biases from Tensorflow

    ```python
    W1_tmp,b1_tmp = layer1.get_weights()
    W2_tmp,b2_tmp = layer2.get_weights()
    W3_tmp,b3_tmp = layer3.get_weights()
    ```

- prediction with the new model
- make a prediction on all the examples at once
- shape of the output

    ```python
    Prediction = my_sequential_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
    Prediction.shape

    # (1000, 1)
    ```

- apply a threshold of 0.5, but to all predictions

    ```python
    Yhat = (Prediction >= 0.5).astype(int)
    print("predict a zero: ",Yhat[0], "predict a one: ", Yhat[500])

    # predict a zero:  [0] predict a one:  [1]
    ```

- make predictions

    ```python
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
        ax.set_axis_off() 
    fig.suptitle("Label, Yhat", fontsize=16)
    plt.show()
    ```

- how a misclassified image looks

Numpy Broadcasting Tutorial

- Z = XW + b utilizes numpy broadcasting to expand the vector b
- XM is a matrix-matrix operation with dimensions (m,j_1)(j1,j_2), which results in a matrix with dimension (m,j_2)
- we add a vector b with dimension (1,j_2)
- b must be expanded to be a (m,j_2) matrix for this element-wise operation to make sense
- the expansion is accomplished by numpy broadcasting
- broadcasting applies to element-wise operations
- basic operation is to ‘stretch’ a smaller dimension by replicating elements to match a larger dimension
- more specifically: when operating on two arrays, numpy compares their shapes element-wise
- it starts with the trailing (rightmost) dimensions and works its way left
- two dimensions are compatible when
    - they are equal, or
    - one of them is 1
- if these conditions are not met, a ValueError: operand could not be broadcast together exception is thrown, indicating that the arrays have incompatible shapes
- the size of the resulting array is the size that is not 1 along each axis of the inputs

- expands along the dimension that is not 1
- guessing that the following code will result in a (3,1) + (1,1) = (3,1) matrix

    ```python
    a = np.array([1,2,3]).reshape(-1,1)  #(3,1)
    b = 5
    print(f"(a + b).shape: {(a + b).shape}, \na + b = \n{a + b}")

    '''
    (a + b).shape: (3, 1), 
    a + b = 
    [[6]
     [7]
     [8]]
    '''
    ```

- applies to all element-wise operations

    ```python
    a = np.array([1,2,3]).reshape(-1,1)  #(3,1)
    b = 5
    print(f"(a * b).shape: {(a * b).shape}, \na * b = \n{a * b}")

    '''
    (a * b).shape: (3, 1), 
    a * b = 
    [[ 5]
     [10]
     [15]]
    '''
    ```

- Row-Column Element-Wise Operations

    ```python
    a = np.array([1,2,3,4]).reshape(-1,1)
    b = np.array([1,2,3]).reshape(1,-1)
    print(a)
    print(b)
    print(f"(a + b).shape: {(a + b).shape}, \na + b = \n{a + b}")

    '''
    [[1]
     [2]
     [3]
     [4]]
    [[1 2 3]]
    (a + b).shape: (4, 3), 
    a + b = 
    [[2 3 4]
     [3 4 5]
     [4 5 6]
     [5 6 7]]
    '''
    ```

- adding a 1D vector b to a (m,j) matrix