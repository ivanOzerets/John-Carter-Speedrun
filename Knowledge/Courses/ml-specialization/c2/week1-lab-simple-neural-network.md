DataSet

- shape of data

    ```python
    X,Y = load_coffee_data();
    print(X.shape, Y.shape)

    # (200, 2) (200, 1)
    ```

- plot the data
- the duration is best kept between 12 and 15 minutes while the temp should be between 175 and 260, as temperature rises, the duration should shrink

Normalize Data

- fitting the weights to the data (back-propagation, covered later) will proceed more quickly if the data is normalized
- the procedure below uses a Keras normalization layer with the following steps
    - create a ‘normalization layer’, not in the model
    - ‘adapt’ the data → leans the mean and variance oft he data set and saves the values internally
    - normalize the data, important to apply normalization to any future data that utilizes the learned model

    ```python
    print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

    '''
    Temperature Max, Min pre normalization: 284.99, 151.32
    Duration    Max, Min pre normalization: 15.45, 11.51
    Temperature Max, Min post normalization: 1.66, -1.69
    Duration    Max, Min post normalization: 1.79, -1.70
    '''
    ```

- tile/copy the data to increase the training set size and reduce the number of training epochs

    ```python
    Xt = np.tile(Xn,(1000,1))
    Yt= np.tile(Y,(1000,1))   
    print(Xt.shape, Yt.shape)   

    # (200000, 2) (200000, 1)
    ```

TensorFlow Model

- Model
    - two layers with sigmoid activations

        ```python
        tf.random.set_seed(1234)  # applied to achieve consistent results
        model = Sequential(
            [
                tf.keras.Input(shape=(2,)),
                Dense(3, activation='sigmoid', name = 'layer1'),
                Dense(1, activation='sigmoid', name = 'layer2')
             ]
        )
        ```

    - Note 1
        - the th.keras.Input(shape=(2,)), specifies the expected shape of the input
        - allows TensorFlow to size the weights and bias parameters at this point
        - useful when exploring TensorFlow models
        - can be omitted in practice and TensorFlow will size the network parameters when he input data is specified in the model.fit statement
    - Note 2
        - including the sigmoid activation in the final layer is not considered best practice
        - it would instead be accounted for in the loss, which improves numerical stability
    - the model.summary() provides a description of the network

        ```python
        model.summary()

        '''
        Model: "sequential"
        _________________________________________________________________
         Layer (type)                Output Shape              Param #   
        =================================================================
         layer1 (Dense)              (None, 3)                 9         

         layer2 (Dense)              (None, 1)                 4         

        =================================================================
        Total params: 13
        Trainable params: 13
        Non-trainable params: 0
        _________________________________________________________________
        '''
        ```

    - the parameter counts shown in the summary correspond to the number of elements in the weight and bias arrays

        ```python
        L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
        L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
        print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

        # L1 params =  9 , L2 params =  4
        ```

    - examine the weights and biases TensorFlow has instantiated
    - the weights W should be of size (number of features in input, number of units in the layer) while the bias b size should match the number of units in the layer:
        - in the first layer with 3 units, we expect W to have a size of (2,3) and b should have 3 elements
        - in the second layer with 1 unit, we expect W to have a size of (3,1) and b should have 1 element

        ```python
        W1, b1 = model.get_layer("layer1").get_weights()
        W2, b2 = model.get_layer("layer2").get_weights()
        print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
        print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

        '''
        W1(2, 3):
         [[ 0.08 -0.3   0.18]
         [-0.56 -0.15  0.89]] 
        b1(3,): [0. 0. 0.]
        W2(3, 1):
         [[-0.43]
         [-0.88]
         [ 0.36]] 
        b2(1,): [0.]
        '''
        ```

    - the following statements will be described in detail later
        - the model.compile statement defines a loss function and specifies a compile optimization
        - the model.fit statement runs GD and fits the weights to the data

        ```python
        model.compile(
            loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
        )

        model.fit(
            Xt,Yt,            
            epochs=10,
        )

        '''
        Epoch 1/10
        6250/6250 [==============================] - 5s 808us/step - loss: 0.1782
        Epoch 2/10
        6250/6250 [==============================] - 5s 832us/step - loss: 0.1165
        Epoch 3/10
        6250/6250 [==============================] - 5s 835us/step - loss: 0.0426
        Epoch 4/10
        6250/6250 [==============================] - 5s 822us/step - loss: 0.0160
        Epoch 5/10
        6250/6250 [==============================] - 5s 832us/step - loss: 0.0104
        Epoch 6/10
        6250/6250 [==============================] - 5s 838us/step - loss: 0.0073
        Epoch 7/10
        6250/6250 [==============================] - 5s 837us/step - loss: 0.0052
        Epoch 8/10
        6250/6250 [==============================] - 5s 841us/step - loss: 0.0037
        Epoch 9/10
        6250/6250 [==============================] - 5s 833us/step - loss: 0.0027
        Epoch 10/10
        6250/6250 [==============================] - 5s 833us/step - loss: 0.0020
        '''
        ```

- Epochs and batches
    - in the fit statement above, the number of epochs was set to 10
    - this specifies that the entire data set should be applied during training 10 times
    - during training, you see the output describing the progress of training
    - the first line, epoch 1/10, describes which epoch the model is currently running
    - for efficiency, the training data set is broken into ‘batches’
    - the default size of a batch in TensorFlow is 32
    - there are 200000 examples in the expanded data set or 6250 batches
    - the notation on the 2nd line 6250/6250 is describing which batch has been executed
- Updated Weights
    - after fitting, the weights have been updated

    ```python
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("W1:\n", W1, "\nb1:", b1)
    print("W2:\n", W2, "\nb2:", b2)

    '''
    W1:
     [[ -0.13  14.3  -11.1 ]
     [ -8.92  11.85  -0.25]] 
    b1: [-11.16   1.76 -12.1 ]
    W2:
     [[-45.71]
     [-42.95]
     [-50.19]] 
    b2: [26.14]
    '''
    ```

    - the values are different from what was printed before calling model.fit()
    - with these, the model should be able to discern what is a good or bad coffee roast
    - instead of using the weights above, first set some weights saved from a previous training run
    - so the notebook remains robust to changes in TensorFlow over time
    - different training runs can produce somewhat different results and the following discussion applies when the model has the weight loaded below

    ```python
    # After finishing the lab later, you can re-run all 
    # cells except this one to see if your trained model
    # gets the same results.

    # Set weights from a previous run. 
    W1 = np.array([
        [-8.94,  0.29, 12.89],
        [-0.17, -7.34, 10.79]] )
    b1 = np.array([-9.87, -9.28,  1.01])
    W2 = np.array([
        [-31.38],
        [-27.86],
        [-32.79]])
    b2 = np.array([15.54])

    # Replace the weights from your trained model with
    # the values above.
    model.get_layer("layer1").set_weights([W1,b1])
    model.get_layer("layer2").set_weights([W2,b2])
    ```

    ```python
    # Check if the weights are successfully replaced
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("W1:\n", W1, "\nb1:", b1)
    print("W2:\n", W2, "\nb2:", b2)

    '''
    W1:
     [[-8.94  0.29 12.89]
     [-0.17 -7.34 10.79]] 
    b1: [-9.87 -9.28  1.01]
    W2:
     [[-31.38]
     [-27.86]
     [-32.79]] 
    b2: [15.54]
    '''
    ```

- Predictions
    - once the model is trained, you can make predictions
    - the output of the model is a probability
    - apply the probability to a threshold → 0.5
    - create input data
    - the model is expecting one or more examples where examples are in the rows of matrix
    - two features so the matrix will be (m,2) where m is the number of examples
    - normalized the test data as well
    - to make a prediction, apply the predict method

        ```python
        X_test = np.array([
            [200,13.9],  # positive example
            [200,17]])   # negative example
        X_testn = norm_l(X_test)
        predictions = model.predict(X_testn)
        print("predictions = \n", predictions)

        '''
        predictions = 
         [[9.63e-01]
         [3.03e-08]]
        '''
        ```

    - to convert the probabilities to a decision, apply the threshold

        ```python
        yhat = np.zeros_like(predictions)
        for i in range(len(predictions)):
            if predictions[i] >= 0.5:
                yhat[i] = 1
            else:
                yhat[i] = 0
        print(f"decisions = \n{yhat}")

        '''
        decisions = 
        [[1.]
         [0.]]
        '''
        ```

    - or more succinctly

        ```python
        yhat = (predictions >= 0.5).astype(int)
        print(f"decisions = \n{yhat}")

        '''
        decisions = 
        [[1]
         [0]]
        '''
        ```

Layer Functions

- examine the functions of the units to determine their role in the coffee roasting decision
- plot the output of each node for all values of the inputs (duration, temp)
- each unit is a logistic function whose output can range from zero to one
- the shading int he graph represent the output value
    - Note: labs typically number things starting at zero rather than 1

    ```python
    plt_layer(X,Y.reshape(-1,),W1,b1,norm_l)
    ```

- the shading shows that each unit is responsible for a different “bad roast”
- unit 0 has larger values when the temperature is too low
- unit 1 has larger values when the duration is too short
- unit 2 has larger values for bad combinations of time/temp
- worth noting that the network leaned these functions on tis own through the process of GD
- same sort of functions a person might choose to make the same decisions
- the function plot of thee final layer is a bit more difficult to visualize
- it’s inputs are the output of the first layer
- the first layer uses sigmoids so their output range is between zero and one
- a 2D plot can be created that calculates the output for all possible combinations of the three inputs
- high output values correspond to ‘bad roast’ areas
- the maximum output is in the areas where the three inputs are small values corresponding to ‘good roast’ areas

    ```python
    plt_output_unit(W2,b2)
    ```

- the final graph shows the whole network in action
- the left graph is the raw output of the final layer represented by the blue shading overlaid on the training data represented by the Xs and Os
- the right graph is the output of the network after a decision threshold
- the Xs and Os correspond to decisions made by the network

    ```python
    netf= lambda x : model.predict(norm_l(x))
    plt_network(X,Y,netf)
    ```