Packages

- Tensorflow is a machine learning package developed by Google
- Google integrated Keras into Tensorflow and released Tensorflow 2.0
- Keras is a framework developed independently by Francois Chollet that creates a simple, layer-centric interface to Tensorflow

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras import Sequential
    from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
    from tensorflow.keras.activations import sigmoid
    from lab_utils_common import dlc
    from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
    plt.style.use('./deeplearning.mplstyle')
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
    ```

Neuron without activation - Regression/Linear Model

- DataSet
    - using linear regression on house prices example

    ```python
    X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
    Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

    fig, ax = plt.subplots(1,1)
    ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
    ax.legend( fontsize='xx-large')
    ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    plt.show()
    ```

- Regression/Linear Model
    - the function implemented by a neuron with no activation is the same as in the first course, linear regression: f_w,b(x^i) = w . x^i + b
    - we can define a layer with one neuron or unit and compare it to the familiar linear regression function

        ```python
        linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
        ```

        ```python
        **linear_layer.get_weights()

        # []**
        ```

    - there are no weights as the weights are not yet instantiated
    - trying the model on one example in X_train will trigger the instantiation of the weights
    - the input to the layer must be 2D, so reshape

        ```python
        a1 = linear_layer(X_train[0].reshape(1,1))
        print(a1)

        # tf.Tensor([[1.67]], shape=(1, 1), dtype=float32)
        ```

    - X_train is a 2D vector, but each value is not
    - X_train[0] is 1D, so reshape to 2D
    - result is a tensor (another name for an array) with a shape of (1,1) or one entry
    - these weights are randomly initialized to small numbers and the bias defaults to being initialized to zero

        ```python
        w, b= linear_layer.get_weights()
        print(f"w = {w}, b={b}")

        # w = [[1.67]], b=[0.]
        ```

    - a linear regression model with a single input feature will have a single weight and bias
    - this matches the dimension of the linear_layer
    - set the weights to known values

        ```python
        set_w = np.array([[200]])
        set_b = np.array([100])

        # set_weights takes a list of numpy arrays
        linear_layer.set_weights([set_w, set_b])
        print(linear_layer.get_weights())

        # [array([[200.]], dtype=float32), array([100.], dtype=float32)]
        ```

    - compare to layer output

        ```python
        a1 = linear_layer(X_train[0].reshape(1,1))
        print(a1)
        alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
        print(alin)

        # tf.Tensor([[300.]], shape=(1, 1), dtype=float32)
        # [[300.]]
        ```

    - predictions can be made on the training data since the values are the same

        ```python
        prediction_tf = linear_layer(X_train)
        prediction_np = np.dot( X_train, set_w) + set_b
        ```

        ```python
        plt_linear(X_train, Y_train, prediction_tf, prediction_np)
        ```

Neuron with Sigmoid activation

- the function implemented by a neuron/unit with a sigmoid activation is the same as the first course, logistic regression

- set w and b to some know values and check the model
- DataSet
    - same example from the first course

        ```python
        X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
        Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
        ```

        ```python
        pos = Y_train == 1 # true of false array
        neg = Y_train == 0

        fig,ax = plt.subplots(1,1,figsize=(4,3))
        ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
        ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
                      edgecolors=dlc["dlblue"],lw=3)

        ax.set_ylim(-0.08,1.1)
        ax.set_ylabel('y', fontsize=12)
        ax.set_xlabel('x', fontsize=12)
        ax.set_title('one variable plot')
        ax.legend(fontsize=12)
        plt.show()
        ```

- Logistic Neuron
    - can implement a ‘logistic neuron’ by adding a sigmoid activation
    - create a Tensorflow Model that contains our logistic layer to demonstrate an alternate method of creating models
    - Tensorflow is often used to create multi-layer models
    - the Sequential model is a convenient means of constructing these models

        ```python
        model = Sequential(
            [
                tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
            ]
        )
        ```

    - model.summary() shows the layers and number of parameters in the model
    - there is only one layer in this model and that layer has only one unit
    - the unit has two parameters, w and b

        ```python
        model.summary()

        '''
        Model: "sequential"
        _________________________________________________________________
         Layer (type)                Output Shape              Param #   
        =================================================================
         L1 (Dense)                  (None, 1)                 2         

        =================================================================
        Total params: 2
        Trainable params: 2
        Non-trainable params: 0
        _________________________________________________________________
        '''
        ```

        ```python
        logistic_layer = model.get_layer('L1')
        w,b = logistic_layer.get_weights()
        print(w,b)
        print(w.shape,b.shape)

        # [[1.21]] [0.]
        # (1, 1) (1,)
        ```

    - set the weight and bias to known values

        ```python
        set_w = np.array([[2]])
        set_b = np.array([-4.5])
        # set_weights takes a list of numpy arrays
        logistic_layer.set_weights([set_w, set_b])
        print(logistic_layer.get_weights())

        # [array([[2.]], dtype=float32), array([-4.5], dtype=float32)]
        ```

    - compare

        ```python
        a1 = model.predict(X_train[0].reshape(1,1))
        print(a1)
        alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
        print(alog)

        # [[0.01]]
        # [[0.01]]
        ```

    - same result so we can use logistic layer and numpy model to make predictions on the training data

        ```python
        plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)
        ```

    - shading reflects the output of the sigmoid which varies from 0 to 1