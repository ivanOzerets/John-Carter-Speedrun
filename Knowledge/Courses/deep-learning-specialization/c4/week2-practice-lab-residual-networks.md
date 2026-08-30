Packages

```python
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from resnets_utils import *
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow

from test_utils import summary, comparator
import public_tests

%matplotlib inline
np.random.seed(1)
tf.random.set_seed(2)
```

The Problem of Very Deep NN

- in recent years, NN have become much deeper, with state-of-the-art networks evolving from having just a few layers (AlexNet) to over a hundred layers
- the main benefit of a very deep network is that it can represent very complex functions
- it can also learn features at many different levels of abstraction, from edges (at the shallower layers, closer to the input) to very complex features (at the deeper layers, closer to the output)
- however, using a deeper network doesn’t always help
- a huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making GD prohibitively slow
- more specifically, during GD, backproping from the final layer back to the first layer, you multiply by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and explode, from gaining very large values)
- during training, you can see the magnitude of the gradient for the shallower layers decrease to zero very rapidly as training proceeds

- solve with Residual Network

Building a Residual Network

- in Resnets, a shortcut or a skip connection allows the model to skip layers

- the image of the left shows the main path through the network
- the image on the right adds a shortcut to the main path
- by stacking these ResNet blocks on top of each other, it forms a very deep network
- also makes it very easy for one of the blocks to learn an identity function
- this means many ResNet blocks can be stacked together with little risk of training set performance
- there is also some evidence that the ease of learning an identity function accounts for ResNets’ remarkable performance even more than skip connections help with vanishing gradients
- the main types of block are used in a ResNet, depending mainly on whether the input/output dines ions are the same or different
- implement both: the identity block and the convolutional block
- The Identity Block
    - the identity block is the standard block used in ResNets, and corresponds to the case where the input activation (a[l]) has the same dimension as the output activation(a[l+2])

    - the upper path is the shortcut path
    - the lower path is the main path
    - notice the conv2d and rely steps in each layer
    - to speed up training, a batchnorm step has been added
    - implement a slightly more powerful version of this identity block, in which the skip connection “skips over” 3 hidden layers rather than 2 layers

    - first component of main path
        - the first conv2d has F1 filters of shape (1,1) and a stride of (1,1)
        - its padding is valid
        - use 0 as the seed for the random uniform initialization: kernel_initializer = initializer(seed=0)
        - the first BatchNorn is normalizing the channels axis
        - then apply the relu activation function → no hypers
    - second component of the main path
        - the second conv2d has F2 filters of shape(f,f) and a stride of (1,1)
        - its padding is same
        - use 0 as the seed for the random uniform initialization: kernel_initializer = initializer(seed=0)
        - the second BatchNorm is normalizing the channels axis
        - then apply the relu activation function → no hypers
    - third component of the main path
        - the third conv2d has F3 filters of shape (1,1) and a stride of (1,1)
        - its padding is valid
        - use 0 as the seed for the random uniform initialization: kernel_initializer = initializer(seed=0)
        - the first BatchNorn is normalizing the channels axis
        - no relu activation function
    - final step
        - the X_shortcut and the output from the 3rd layer X are added together
        - Add()([var,var2])
        - then apply the relu activation → no hypers
    - Exercise 1 - identity_block
        - implement the ResNet identity block
        - to implement the conv2D step: Conv2D
        - to implement BatchNorm: BatchNormalization(axis = 3)(X)
        - if training is set to false, its weights are not updates with the new example, i.e. when the model is used in prediction mode
        - for the activation use Activation(’relu’)(X)
        - to add the value passed forward by the shortcut: Add
        - the initializer argument is a parameter that receives an initializer function which by default is random_uniform
        - the functions accept a seed argument that can be any value → set to zero here

        ```python
        # UNQ_C1
        # GRADED FUNCTION: identity_block

        def identity_block(X, f, filters, initializer=random_uniform):
            """
            Implementation of the identity block as defined in Figure 4

            Arguments:
            X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -- integer, specifying the shape of the middle CONV's window for the main path
            filters -- python list of integers, defining the number of filters in the CONV layers of the main path
            initializer -- to set up the initial weights of a layer. Equals to random uniform initializer

            Returns:
            X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
            """

            # Retrieve Filters
            F1, F2, F3 = filters

            # Save the input value. You'll need this later to add back to the main path. 
            X_shortcut = X

            # First component of main path
            X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
            X = BatchNormalization(axis = 3)(X) # Default axis
            X = Activation('relu')(X)

            ### START CODE HERE
            ## Second component of main path (≈3 lines)
            ## Set the padding = 'same'
            X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
            X = BatchNormalization(axis = 3)(X)
            X = Activation('relu')(X)

            ## Third component of main path (≈2 lines)
            ## Set the padding = 'valid'
            X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
            X = BatchNormalization(axis = 3)(X)

            ## Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
            X = Add()([X, X_shortcut])
            X = Activation('relu')(X)
            ### END CODE HERE

            return X

        '''
        With training=False

        [[[  0.        0.        0.        0.     ]
          [  0.        0.        0.        0.     ]]

         [[192.99992 192.99992 192.99992  96.99996]
          [ 96.99996  96.99996  96.99996  48.99998]]

         [[578.99976 578.99976 578.99976 290.99988]
          [290.99988 290.99988 290.99988 146.99994]]]
        96.99996

        With training=True

        [[[0.      0.      0.      0.     ]
          [0.      0.      0.      0.     ]]

         [[0.40732 0.40732 0.40732 0.40732]
          [0.40732 0.40732 0.40732 0.40732]]

         [[5.00011 5.00011 5.00011 3.25955]
          [3.25955 3.25955 3.25955 2.40732]]]
        '''
        ```

- The Convolutional Block
    - the ResNet “convolutional block” is the second block type
    - use this block type when the input and output dimensions don’t match up
    - the difference with the identify block is that there is a conv2d layer in the shortcut path

    - the conv2d layer in the shortcut path is used to resize the input x to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path (similar role to Ws)
    - for example, to reduce the activation dimensions’ height and width by a factor of 2, use a 1c1 convolution with a stride of 2
    - the con2d layer on the shortcut path does not use any non-linear activation function
    - its main role is to just apply a learned linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step
    - the additional initializer argument is required for grading purposes, adn it has been set by default to glorot_uniform
    - first component of the main path
        - the first conv2d has F1 filters of shape (1,1) and a stride of (s,s)
        - its padding is valid
        - use 0 as the glorot_uniform seed kernel_initializer = initializer(seed=0)
        - the first BatchNorm is normalizing the channels axis
        - then apply the relu activation function → no hypers
    - second component of the main path
        - the second conv2d has F2 filters of shape (f,f) and a stride of (1,1)
        - its padding is same
        - use 0 as the glorot_uniform seed kernel_initializer = initializer(seed=0)
        - the second BatchNorm is normalizing the channels axis
        - then apply the relu activation function → no hypers
    - third component
        - the third conv2d has F3 filters of shape (1,1) and a stride of (1,1)
        - its padding is valid
        - use 0 as the glorot_uniform seed kernel_initializer = initializer(seed=0)
        - the first BatchNorm is normalizing the channels axis
        - no relu
    - shortcut path
        - the conv2d has F3 filters of shape (1,1) and a stride of (s,s)
        - its padding is valid
        - use 0 as the glorot_uniform seed kernel_initializer = initializer(seed=0)
        - the first BatchNorm is normalizing the channels axis
    - final step
        - the shortcut and the main path values are added together
        - then apply relu
    - Exercise 2 - convolutional_block
        - implement the convolutional block
        - glorot_uniform initializer

            ```python
            # UNQ_C2
            # GRADED FUNCTION: convolutional_block

            def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
                """
                Implementation of the convolutional block as defined in Figure 4

                Arguments:
                X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
                f -- integer, specifying the shape of the middle CONV's window for the main path
                filters -- python list of integers, defining the number of filters in the CONV layers of the main path
                s -- Integer, specifying the stride to be used
                initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                               also called Xavier uniform initializer.

                Returns:
                X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
                """

                # Retrieve Filters
                F1, F2, F3 = filters

                # Save the input value
                X_shortcut = X

                ##### MAIN PATH #####

                # First component of main path glorot_uniform(seed=0)
                X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
                X = BatchNormalization(axis = 3)(X)
                X = Activation('relu')(X)

                ### START CODE HERE

                ## Second component of main path (≈3 lines)
                X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding='same', kernel_initializer = initializer(seed=0))(X)
                X = BatchNormalization(axis = 3)(X)
                X = Activation('relu')(X)

                ## Third component of main path (≈2 lines)
                X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding='valid', kernel_initializer = initializer(seed=0))(X)
                X = BatchNormalization(axis = 3)(X)

                ##### SHORTCUT PATH ##### (≈2 lines)
                X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
                X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

                ### END CODE HERE

                # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
                X = Add()([X, X_shortcut])
                X = Activation('relu')(X)

                return X

            '''
            tf.Tensor(
            [[[0.33485505 1.6415989  0.33789736 0.08511472 0.814965   0.        ]
              [0.17509979 1.5699672  0.2606045  0.         0.767209   0.        ]]

             [[0.         1.4983511  0.16896994 0.         0.61830646 0.        ]
              [0.         1.4502985  0.11632714 0.         0.58068544 0.        ]]], shape=(2, 2, 6), dtype=float32)
            '''
            ```

Building Your First ResNet Model (50 layers)

- now have the necessary blocks to build a very deep ResNet
- the “ID BLOCK” in the diagram stands for identity block and “ID BLOCK x3” means to stack 3 identity blocks together

- zero-padding pads the input with a pad of (3,3)
- stage 1
    - the 2d conv has 64 filters of shape (7,7) and uses a stride of (2,2)
    - batchnorm is applied to the channels of the input
    - maxpooling uses a (3,3) window and a (2,2) stride
- stage 2
    - the conv block uses three sets of filters of size [64, 64, 256], f is 3, and s is 1
    - the 2 ID blocks use three sets of filters of size [64, 64, 256], and f is 3
- stage 3
    - the conv block uses three sets of filters of size [128, 128, 512], f is 3, and s is 2
    - the 3 ID blocks use three sets of filters of size [128, 128, 512], and f is 3
- stage 4
    - the conv block uses three sets of filters of size [256, 256, 1024], f is 3, and s is 2
    - the 5 ID blocks use three sets of filters of size [256, 256, 1024], and f is 3
- stage 5
    - the conv block uses three sets of filters of size [512, 512, 2048], f is 3, and s is 2
    - the 2 ID blocks use three sets of filters of size [512, 512, 2048], and f is 3
- the 2d average pooling uses a window (pool_size) of shape (2,2)
- the flatten layer doesn’t have any hypers
- the fully connected (dense) layer reduces its input to the number of classes using a softmax activation
- Exercise 3 - ResNet50
    - implement the ResNet with 50 layers

        ```python
        # UNQ_C3
        # GRADED FUNCTION: ResNet50

        def ResNet50(input_shape = (64, 64, 3), classes = 6, training=False):
            """
            Stage-wise implementation of the architecture of the popular ResNet50:
            CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
            -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

            Arguments:
            input_shape -- shape of the images of the dataset
            classes -- integer, number of classes

            Returns:
            model -- a Model() instance in Keras
            """

            # Define the input as a tensor with shape input_shape
            X_input = Input(input_shape)

            # Zero-Padding
            X = ZeroPadding2D((3, 3))(X_input)

            # Stage 1
            X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = 3)(X)
            X = Activation('relu')(X)
            X = MaxPooling2D((3, 3), strides=(2, 2))(X)

            # Stage 2
            X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
            X = identity_block(X, 3, [64, 64, 256])
            X = identity_block(X, 3, [64, 64, 256])

            ### START CODE HERE

            # Use the instructions above in order to implement all of the Stages below
            # Make sure you don't miss adding any required parameter

            ## Stage 3 (≈4 lines)
            # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
            X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)

            # the 3 `identity_block` with correct values of `f` and `filters` for this stage
            X = identity_block(X, 3, [128, 128, 512])
            X = identity_block(X, 3, [128, 128, 512])
            X = identity_block(X, 3, [128, 128, 512])

            # Stage 4 (≈6 lines)
            # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
            X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)

            # the 5 `identity_block` with correct values of `f` and `filters` for this stage
            X = identity_block(X, 3, [256, 256, 1024])
            X = identity_block(X, 3, [256, 256, 1024])
            X = identity_block(X, 3, [256, 256, 1024])
            X = identity_block(X, 3, [256, 256, 1024])
            X = identity_block(X, 3, [256, 256, 1024])

            # Stage 5 (≈3 lines)
            # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
            X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)

            # the 2 `identity_block` with correct values of `f` and `filters` for this stage
            X = identity_block(X, 3, [512, 512, 2048])
            X = identity_block(X, 3, [512, 512, 2048])

            # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
            X = AveragePooling2D(pool_size=(2, 2))(X)

            ### END CODE HERE

            # output layer
            X = Flatten()(X)
            X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)

            # Create model
            model = Model(inputs = X_input, outputs = X)

            return model
        ```

        ```python
        tf.keras.backend.set_learning_phase(True)

        model = ResNet50(input_shape = (64, 64, 3), classes = 6)
        print(model.summary())

        '''
        Model: "model"
        __________________________________________________________________________________________________
         Layer (type)                   Output Shape         Param #     Connected to                     
        ==================================================================================================
         input_5 (InputLayer)           [(None, 64, 64, 3)]  0           []                               

         zero_padding2d_4 (ZeroPadding2  (None, 70, 70, 3)   0           ['input_5[0][0]']                
         D)                                                                                               

         conv2d_172 (Conv2D)            (None, 32, 32, 64)   9472        ['zero_padding2d_4[0][0]']       

         batch_normalization_172 (Batch  (None, 32, 32, 64)  256         ['conv2d_172[0][0]']             
         Normalization)                                                                                   

         activation_149 (Activation)    (None, 32, 32, 64)   0           ['batch_normalization_172[0][0]']

         max_pooling2d_4 (MaxPooling2D)  (None, 15, 15, 64)  0           ['activation_149[0][0]']         
         ...
          add_65 (Add)                   (None, 2, 2, 2048)   0           ['batch_normalization_224[0][0]',
                                                                          'activation_194[0][0]']         

         activation_197 (Activation)    (None, 2, 2, 2048)   0           ['add_65[0][0]']                 

         average_pooling2d (AveragePool  (None, 1, 1, 2048)  0           ['activation_197[0][0]']         
         ing2D)                                                                                           

         flatten (Flatten)              (None, 2048)         0           ['average_pooling2d[0][0]']      

         dense (Dense)                  (None, 6)            12294       ['flatten[0][0]']                

        ==================================================================================================
        Total params: 23,600,006
        Trainable params: 23,546,886
        Non-trainable params: 53,120
        __________________________________________________________________________________________________
        None
        '''
        ```

    - prior to training a model, configure the learning process by compiling the model

        ```python
        np.random.seed(1)
        tf.random.set_seed(2)
        opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        ```

    - use the SIGNS dataset

        ```python
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

        # Normalize image vectors
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.

        # Convert training and test labels to one hot matrices
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

    - train the model on 10 epochs with a batch size of 32

        ```python
        model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

        '''
        Epoch 1/10
        34/34 [==============================] - 13s 58ms/step - loss: 1.8084 - accuracy: 0.2898
        Epoch 2/10
        34/34 [==============================] - 1s 41ms/step - loss: 1.2374 - accuracy: 0.5083
        Epoch 3/10
        34/34 [==============================] - 1s 41ms/step - loss: 0.9176 - accuracy: 0.6565
        Epoch 4/10
        34/34 [==============================] - 1s 40ms/step - loss: 0.5906 - accuracy: 0.7694
        Epoch 5/10
        34/34 [==============================] - 1s 40ms/step - loss: 0.4196 - accuracy: 0.8398
        Epoch 6/10
        34/34 [==============================] - 1s 41ms/step - loss: 0.2638 - accuracy: 0.9083
        Epoch 7/10
        34/34 [==============================] - 1s 42ms/step - loss: 0.2316 - accuracy: 0.9130
        Epoch 8/10
        34/34 [==============================] - 1s 42ms/step - loss: 0.1462 - accuracy: 0.9481
        Epoch 9/10
        34/34 [==============================] - 1s 40ms/step - loss: 0.2037 - accuracy: 0.9352
        Epoch 10/10
        34/34 [==============================] - 1s 40ms/step - loss: 0.1592 - accuracy: 0.9435

        '''
        ```

    - how does the model perform on the test set

        ```python
        preds = model.evaluate(X_test, Y_test)
        print ("Loss = " + str(preds[0]))
        print ("Test Accuracy = " + str(preds[1]))

        '''
        4/4 [==============================] - 4s 10ms/step - loss: 0.5465 - accuracy: 0.8250
        Loss = 0.5464782118797302
        Test Accuracy = 0.824999988079071
        '''
        ```

    - only trained for 10 epochs
    - using a pretrained model

        ```python
        pre_trained_model = load_model('resnet50.h5')

        preds = pre_trained_model.evaluate(X_test, Y_test)
        print ("Loss = " + str(preds[0]))
        print ("Test Accuracy = " + str(preds[1]))

        '''
        4/4 [==============================] - 1s 8ms/step - loss: 0.1596 - accuracy: 0.9500
        Loss = 0.15959423780441284
        Test Accuracy = 0.949999988079071
        '''
        ```

Testing on Your Own Image

```python
img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0
x2 = x 
print('Input image shape:', x.shape)
imshow(img)
prediction = pre_trained_model.predict(x2)
print("Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ", prediction)
print("Class:", np.argmax(prediction))
```