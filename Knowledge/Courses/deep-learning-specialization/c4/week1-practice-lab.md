Packages

```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

- Load the Data and Split the Data into Train/Test Sets
    - using the Happy House dataset
    - contains images of peoples’ faces
    - tasked to build a ConvNet that determines whether the people in the images are smiling or not

        ```python
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

        # Normalize image vectors
        X_train = X_train_orig/255.
        X_test = X_test_orig/255.

        # Reshape
        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T

        print ("number of training examples = " + str(X_train.shape[0]))
        print ("number of test examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_test shape: " + str(X_test.shape))
        print ("Y_test shape: " + str(Y_test.shape))

        '''
        number of training examples = 600
        number of test examples = 150
        X_train shape: (600, 64, 64, 3)
        Y_train shape: (600, 1)
        X_test shape: (150, 64, 64, 3)
        Y_test shape: (150, 1)
        '''
        ```

    - display the images contained in the dataset images are 64x64 pixels in RGB format

        ```python
        index = 124
        plt.imshow(X_train_orig[index]) #display sample training image
        plt.show()
        ```

Layers in TF Keras

- layers were previously created manually in munpy
- in TF Keras, no code is written to directly create layers
- rather, TF Keras has pre-defined layers to use
- when create a layer in TF Keras, a function is created that takes some input and transforms it into an output to reuse later

The Sequential API

- previously, helper functions were built using numpy to understand the mechanics behind convolutional neural networks
- most practical applications of deep learning today are build using programing frameworks, which have many build-in function to call
- Keras is a high-level abstraction build on top of TenorFLow, which allows for even more simplified an optimized model creation and training
- create a model using TF Keras’ Sequential API, which allows to build layer by layer, and is ideal for building models where each layer has exactly one input tensor and one output tensor
- using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks
- Funcational API is more flexible, powerful alternative
- Create the Sequential Model
    - the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order
    - also possible to add layers incrementally to a Sequential model with the .add() method, or remove them using the .pop() method
    - a Sequential model can be thought of as behaving like a list of layers
    - like python lists, Sequential layers are ordered, and the order in which they are specified matters
    - if the model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn’t be the right choice
    - for any layer construction in Keras, specify the input shape in advance
    - this is because in Keras, the shape of the weights is based on the shape of the inputs
    - the weights are only created when the model first sees some input data
    - Sequential models can be crated by passing a list of layers to the Sequential constructor
    - Exercise 1 - happyModel
        - implement the happyModel function to build the following model
        - ZEROPAD2D → CONV2D → BATCHNORM → RELU → MAXPOOL → FLATTEN → DENSE
        - plug in the following parameters for all the steps
            - ZeroPadding2ED: padding 3, input shape 64x64x3
            - Conv2D: use 32 7x7 filters, stride 1
            - BatchNormalization: for axis 3
            - ReLU
            - MaxPool2D: using default parameters
            - Flatten the previous output
            - Fully-connected (Dense) layer: apply a fully connected layer with 1 neuron and a sigmoid activation
            - tfl is shorthand for tensorflow.keras.layers

        ```python
        # GRADED FUNCTION: happyModel

        def happyModel():
            """
            Implements the forward propagation for the binary classification model:
            ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

            Note that for simplicity and grading purposes, you'll hard-code all the values
            such as the stride and kernel (filter) sizes. 
            Normally, functions should take these values as function parameters.

            Arguments:
            None

            Returns:
            model -- TF Keras model (object containing the information for the entire training process) 
            """
            model = tf.keras.Sequential([
                    ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3

                    ## Conv2D with 32 7x7 filters and stride of 1

                    ## BatchNormalization for axis 3

                    ## ReLU

                    ## Max Pooling 2D with default parameters

                    ## Flatten layer

                    ## Dense layer with 1 unit for output & 'sigmoid' activation

                    # YOUR CODE STARTS HERE
                    tfl.Input(shape=(64,64,3)),
                    tfl.ZeroPadding2D(padding=3),
                    tfl.Conv2D(filters=32, kernel_size=7, strides=1),
                    tfl.BatchNormalization(axis=3),
                    tfl.ReLU(),
                    tfl.MaxPool2D(),
                    tfl.Flatten(),
                    tfl.Dense(units=1, activation="sigmoid")
                    # YOUR CODE ENDS HERE
                ])

            return model

        '''
        ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
        ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
        ['BatchNormalization', (None, 64, 64, 32), 128]
        ['ReLU', (None, 64, 64, 32), 0]
        ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
        ['Flatten', (None, 32768), 0]
        ['Dense', (None, 1), 32769, 'sigmoid']
        '''
        ```

    - now that the model is created, compile it for training with an optimizer and a loss
    - when the string accuracy is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used
    - an optimization build into TensorFlow

        ```python
        happy_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        ```

    - check the model’s parameters with the .summary() method
    - displays the types of layers present, the shape of the outputs, and how many parameters are in each layers

        ```python
        happy_model.summary()

        '''
        Model: "sequential_6"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        zero_padding2d_11 (ZeroPaddi (None, 70, 70, 3)         0         
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 64, 64, 32)        4736      
        _________________________________________________________________
        batch_normalization_4 (Batch (None, 64, 64, 32)        128       
        _________________________________________________________________
        re_lu_4 (ReLU)               (None, 64, 64, 32)        0         
        _________________________________________________________________
        max_pooling2d_4 (MaxPooling2 (None, 32, 32, 32)        0         
        _________________________________________________________________
        flatten_4 (Flatten)          (None, 32768)             0         
        _________________________________________________________________
        dense_4 (Dense)              (None, 1)                 32769     
        =================================================================
        Total params: 37,633
        Trainable params: 37,569
        Non-trainable params: 64
        _________________________________________________________________

        '''
        ```

- Train and Evaluate the Model
    - simply call .fit() to train
    - no need for min-batching, saving, or complex backprop computations
    - batches are specified already
    - epoch number or minibatch size is an option to specify

        ```python
        happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)

        '''
        Epoch 1/10
        38/38 [==============================] - 4s 102ms/step - loss: 1.2239 - accuracy: 0.6900
        ...
        Epoch 10/10
        38/38 [==============================] - 4s 95ms/step - loss: 0.0676 - accuracy: 0.9817

        '''
        ```

    - after completion, use .evaluate() to evaluate against the test set
    - this function will print the value of the loss function and the performance metrics specified during the compilation of the model
    - in this case, the binary_crossentropy and the accuracy

        ```python
        happy_model.evaluate(X_test, Y_test)

        '''
        5/5 [==============================] - 0s 27ms/step - loss: 0.2864 - accuracy: 0.8533

        [0.2863641679286957, 0.8533333539962769]
        '''
        ```

    - what if the model needs shared layers, branches, or multiple inputs and outputs

The Functional API

- build a ConvNet that can differentiate between six sign language digits
- the functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs
- imagine that, where the sequential api requires the model to move in a linear fashion through its layers, the function api allows much more flexibility
- where sequential is a straight line, a functional model is a graph, where the nodes of the layers can connect in many more ways than one
- in the visual, the one possible direction of the movement sequential model is shown in contrast to a skip connection, which is just one of the manty ways a functional model can be constructed
- a skip connection, skips some layer in the network and feeds the output to a later layer in the network

- Load the SIGNS Dataset
    - the SIGNS dataset is a collection of six signs representing numbers from 0 to 5

        ```python
        # Loading the data (signs)
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
        ```

    - an example of a labelled image in the dataset

        ```python
        # Example of an image from the dataset
        index = 9
        plt.imshow(X_train_orig[index])
        print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
        ```

- Split the Data into Train/Test Sets
    - more natural to apply ConvNet

        ```python
        X_train = X_train_orig/255.
        X_test = X_test_orig/255.
        Y_train = convert_to_one_hot(Y_train_orig, 6).T
        Y_test = convert_to_one_hot(Y_test_orig, 6).T
        print ("number of training examples = " + str(X_train.shape[0]))
        print ("number of test examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_test shape: " + str(X_test.shape))
        print ("Y_test shape: " + str(Y_test.shape))

        '''
        number of training examples = 1080
        number of test examples = 120
        X_train shape: (1080, 64, 64, 3)
        Y_train shape: (1080, 6)
        X_test shape: (120, 64, 64, 3)
        Y_test shape: (120, 6)
        '''
        ```

- Forward Propagation
    - in tf, there are built-in functions that implement the convolution steps
    - the functional api creates a graph of layers
    - the following model could also be defined using the sequential api since the information flow is on a single line
    - being building the graph of layers by creating an input node that functions as a callable object
        - input_img = tf.keras.Input(shape=input_shape):
    - then, create a new node in the graph of layers by calling a layer on the input_img object
        - tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img)
        - tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'): downsamples the input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window
            - for max pooling, usually operate on a single example at a time and a single channel at a time
        - tf.keras.layers.ReLU(): computes the elementwise ReLU of Z (any shape)
        - tf.keras.layers.Flatten(): given a tensor P, this function takes each training (or test) example in the batch and flattens it into a 1D vector
            - if a tensor P has the shape (batch_size, h, w, c), it returns a flattened tensor with shape (batch_size, k), where k = h * w * c, k equals the product of all the dimension sizes other than the first dimension
        - tf.keras.layers.Dense(units= ... , activation='softmax')(F): given the flattened input F, it returns the output computed using a fully connected layer
    - in the last function, the fully connected layer automatically initializes weights in the graph and keeps on training them as the model is training
    - no need to initialize the weights
    - before creating the model, define the outptut using the last of ht function’s compositions
        - outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)
    - window, kernel, filter, pool
        - the words kernel and filter are used to refer to the same thing
        - the word filter accounts for the amount of kernels that will be used in a single convolution layer
        - pool is the name of the operation that takes the max or average values of the kernels
        - this is why the parameter pool_size refers to kernel_sizea nd use (f, f( to refer to the filter size
        - pool size and kernel size refer to the same thing in different objects - they refer to the shape of the window where the operation takes place
    - Exercise 2 - convolutional_model
        - implement the convolutional_model function to build the following model
        - CONV2D → RELU → MAXPOOL → CONV2D → RELU → MAXPOOL → FLATTEN → DENSE
        - plug in the following parameters for all the steps
            - Conv2D: use 8 4x4 filters, stride 1, padding is “SAME”
            - ReLU
            - MaxPool2D: use an 8x8 filter size and an 8x8 stride, padding is “SAME”
            - Conv2D: use 16 2x2 filters, stride 1, padding is “SAME”
            - ReLU
            - MaxPool2D: use 4x4 filter size and a 4x4 stride, padding is “SAME”
            - Flatten the previous output
            - fully-connected layer: apply a fully connected layer with six neurons and a softmax activation

            ```python
            # GRADED FUNCTION: convolutional_model

            def convolutional_model(input_shape):
                """
                Implements the forward propagation for the model:
                CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

                Note that for simplicity and grading purposes, you'll hard-code some values
                such as the stride and kernel (filter) sizes. 
                Normally, functions should take these values as function parameters.

                Arguments:
                input_img -- input dataset, of shape (input_shape)

                Returns:
                model -- TF Keras model (object containing the information for the entire training process) 
                """

                input_img = tf.keras.Input(shape=input_shape)
                ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
                # Z1 = None
                ## RELU
                # A1 = None
                ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
                # P1 = None
                ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
                # Z2 = None
                ## RELU
                # A2 = None
                ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
                # P2 = None
                ## FLATTEN
                # F = None
                ## Dense layer
                ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
                # outputs = None
                # YOUR CODE STARTS HERE
                Z1 = tfl.Conv2D(filters=8, kernel_size=4, strides=1, padding='same')(input_img)
                A1 = tfl.ReLU()(Z1)
                P1 = tfl.MaxPool2D(pool_size=8, strides=8, padding='same')(A1)
                Z2 = tfl.Conv2D(filters=16, kernel_size=2, padding='same')(P1)
                A2 = tfl.ReLU()(Z2)
                P2 = tfl.MaxPool2D(pool_size=4, strides=4, padding='same')(A2)
                F = tfl.Flatten()(P2)
                outputs = tfl.Dense(units=6, activation='softmax')(F)
                # YOUR CODE ENDS HERE
                model = tf.keras.Model(inputs=input_img, outputs=outputs)
                return model

            '''
            Model: "functional_2"
            _________________________________________________________________
            Layer (type)                 Output Shape              Param #   
            =================================================================
            input_6 (InputLayer)         [(None, 64, 64, 3)]       0         
            _________________________________________________________________
            conv2d_8 (Conv2D)            (None, 64, 64, 8)         392       
            _________________________________________________________________
            re_lu_7 (ReLU)               (None, 64, 64, 8)         0         
            _________________________________________________________________
            max_pooling2d_7 (MaxPooling2 (None, 8, 8, 8)           0         
            _________________________________________________________________
            conv2d_9 (Conv2D)            (None, 8, 8, 16)          528       
            _________________________________________________________________
            re_lu_8 (ReLU)               (None, 8, 8, 16)          0         
            _________________________________________________________________
            max_pooling2d_8 (MaxPooling2 (None, 2, 2, 16)          0         
            _________________________________________________________________
            flatten_4 (Flatten)          (None, 64)                0         
            _________________________________________________________________
            dense_4 (Dense)              (None, 6)                 390       
            =================================================================
            Total params: 1,310
            Trainable params: 1,310
            Non-trainable params: 0
            _________________________________________________________________
            '''
            ```

        - both the sequential and function apis return a tf keras model object
        - the only difference is how inputs are handled inside the object model
- Train the Model

    ```python
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
    history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)

    '''
    Epoch 1/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.8224 - accuracy: 0.1667 - val_loss: 1.7908 - val_accuracy: 0.1667
    Epoch 2/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.7876 - accuracy: 0.1639 - val_loss: 1.7833 - val_accuracy: 0.1500
    ...
    Epoch 99/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3738 - accuracy: 0.8898 - val_loss: 0.5203 - val_accuracy: 0.7667
    Epoch 100/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3704 - accuracy: 0.8926 - val_loss: 0.5174 - val_accuracy: 0.7667
    '''
    ```

History Object

- the history object is an output of the .fit() operation, and provides a record of all the loss metric values in memory
- it’s stored as a dictionary to retrieve at history.history

    ```python
    history.history

    '''
    {'loss': [1.8224058151245117,
      1.7876347303390503,
      ...
      0.7666666507720947,
      0.7666666507720947]}
    '''
    ```

- now visualize the loss over time using history.history

    ```python
    # The history.history["loss"] entry is a dictionary with as many values as epochs that the
    # model was trained on. 
    df_loss_acc = pd.DataFrame(history.history)
    df_loss= df_loss_acc[['loss','val_loss']]
    df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    df_acc= df_loss_acc[['accuracy','val_accuracy']]
    df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    ```