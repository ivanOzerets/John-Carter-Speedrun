Packages

```python
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time
```

Basic Optimization with GradientTape

- the beauty of TensorFlow 2 is in its simplicity
- just implement the forward prop through a computational graph
- TensorFlow will compute the derivatives, by moving backwards through the graph recorded with GradientTape
- then specify the cost function and optimizer
- when writing a TensorFlow program, the main object to get used and trasnformed is the tf.Tensor
- these tensors are the TensorFlow equivalent of Numpy array, i.e. multidimensional arrays of a given data type that also contain information about the computational graph
- use tf.Varaible to store the state of the variables
- variables can only be created once as its initial values defines the variable shape and type
- additionally, the dtype arg in tf.Variable can be set to allow data to be converted to that type
- if none is specified, either the datatype will be kept if the initial value is a Tensor, or conver_to_tensor will decide
- best to specify directly
- call teh TensorFlow dataset created on a HDF5 file, which can be used in place of a numpy array to store the datasets
- can be thought of as TensorFlow data generator
- use the hand sign data set, that is composed of images with shape 64x64x3

    ```python
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    ```

    ```python
    x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])
    ```

    ```python
    type(x_train)

    # tensorflow.python.data.ops.dataset_ops.TensorSliceDataset
    ```

- TensorFlow Datasets are generators, so you can’t access directly the contents unless iterated over in a loop, or by explicitly creating a python iterator using iter and consuming its elements using next
- also, inspect the shape and dtype of each element using the element spec attribute

    ```python
    print(x_train.element_spec)

    # TensorSpec(shape=(64, 64, 3), dtype=tf.uint8, name=None)
    ```

    ```python
    print(next(iter(x_train)))

    # tf.Tensor(
    [[[227 220 214]
      [227 221 215]
      [227 222 215]
      ...
      [209 210 210]
      [209 210 209]
      [208 209 209]]], shape=(64, 64, 3), dtype=uint8)
    ```

- the dataset is a subset of the sign language digits
- it contains six different classes representing the digits from 0 to 5

    ```python
    unique_labels = set()
    for element in y_train:
        unique_labels.add(element.numpy())
    print(unique_labels)

    # {0, 1, 2, 3, 4, 5}
    ```

- see some of the images int eh dataset

    ```python
    images_iter = iter(x_train)
    labels_iter = iter(y_train)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(next(images_iter).numpy().astype("uint8"))
        plt.title(next(labels_iter).numpy().astype("uint8"))
        plt.axis("off")
    ```

- one more difference between TensorFlow datasets and Numpy arrays: to transform a dataset, invoke the map method to apply the function passed as an argument to each of the elements

    ```python
    def normalize(image):
        """
        Transform an image into a tensor of shape (64 * 64 * 3, )
        and normalize its components.

        Arguments
        image - Tensor.

        Returns: 
        result -- Transformed tensor 
        """
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [-1,])
        return image
    ```

    ```python
    new_train = x_train.map(normalize)
    new_test = x_test.map(normalize)
    ```

    ```python
    new_train.element_spec

    # TensorSpec(shape=(12288,), dtype=tf.float32, name=None)
    ```

    ```python
    print(next(iter(new_train)))

    # tf.Tensor([0.8901961  0.8627451  0.8392157  ... 0.8156863  0.81960785 0.81960785], shape=(12288,), dtype=float32)
    ```

- Linear Function
    - begin by computing the following equation Y = WX + b, where W and X are random matrices and b is a random vector
    - Exercise 1  - linear_function
        - Compute WX + b where W, X, and b are drawn from a random normal distribution
        - W is of shape (4,3), X is (3,1) and b is (4,1)
        - this is how to define a constant X with the shape (3,1)
            - X = tf.constant(np.random.randn(3,1), name = “X”)
        - the different between tf.constant and tf.Variable is the state of a tf.Variable can be modified but the state of a tf.constant cannot
        - taf.matmul(…, …) for matrix mult
        - tf.add(…, …) to add
        - np.random.rand(…) for random initialization

        ```python
        # GRADED FUNCTION: linear_function

        def linear_function():
            """
            Implements a linear function: 
                    Initializes X to be a random tensor of shape (3,1)
                    Initializes W to be a random tensor of shape (4,3)
                    Initializes b to be a random tensor of shape (4,1)
            Returns: 
            result -- Y = WX + b 
            """

            np.random.seed(1)

            """
            Note, to ensure that the "random" numbers generated match the expected results,
            please create the variables in the order given in the starting code below.
            (Do not re-arrange the order).
            """
            # (approx. 4 lines)
            # X = ...
            # W = ...
            # b = ...
            # Y = ...
            # YOUR CODE STARTS HERE
            X = tf.constant(np.random.randn(3,1), name = 'X')
            W = tf.constant(np.random.randn(4,3), name = 'W')
            b = tf.constant(np.random.randn(4,1), name = 'Y')
            Y = tf.add(tf.matmul(W, X), b)
            # YOUR CODE ENDS HERE
            return Y

        '''
        tf.Tensor(
        [[-2.15657382]
         [ 2.95891446]
         [-1.08926781]
         [-0.84538042]], shape=(4, 1), dtype=float64)
        '''
        ```

- Computing the Sigmoid
    - TensorFlow offers a variety of commonly used network function like tf.sigmoid and tf.softmax
    - compute the sigmoid of z
    - cast tensor to type float32 using tf.cast
    - then compute the sigmoid using tf.keras.activations.sigmoid
    - Exercise 2 - sigmoid
        - implement the sigmoid function
            - tf.cast(…, tf.float32)
            - tf.keras.activations.sigmoid(…)

        ```python
        # GRADED FUNCTION: sigmoid

        def sigmoid(z):

            """
            Computes the sigmoid of z

            Arguments:
            z -- input value, scalar or vector

            Returns: 
            a -- (tf.float32) the sigmoid of z
            """
            # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.

            # (approx. 2 lines)
            # z = ...
            # a = ...
            # YOUR CODE STARTS HERE
            z = tf.cast(z, tf.float32)
            a = tf.keras.activations.sigmoid(z)
            # YOUR CODE ENDS HERE
            return a

        ```

- Using One Hot Encodings
    - many times in deep learning, the Y vector with numbers ranging from 0 to C - 1, where C is the number of classes
    - if C is for example 4, then the following y vector needs to be converted

    - this is called one hot encoding, because in the converted representation, exactly one element of each column is “hot”
    - to do this conversion in numpy, there will be a few lines
    - in TensorFlow: tf.one_hot(labels, depth, axis=0)
    - Exercise 3 - one_hot_matrix
        - implement the function to take one label and total number of classes C, adn return the one hot encoding in a one-dimensional rensor
        - use tf.ont_hot() and tf.reshape() to reshape the one hot tensor: tf.reshape(tensor, shape)

        ```python
        # GRADED FUNCTION: one_hot_matrix
        def one_hot_matrix(label, C=6):
            """
            Computes the one hot encoding for a single label

            Arguments:
                label --  (int) Categorical labels
                C --  (int) Number of different classes that label can take

            Returns:
                 one_hot -- tf.Tensor A one-dimensional tensor (array) with the one hot encoding.
            """
            # (approx. 1 line)
            # one_hot = None(None(None, None, None), shape=[C, ]) 
            # YOUR CODE STARTS HERE
            one_hot = tf.reshape(tf.one_hot(label, C, axis=0), shape=[C, ])  
            # YOUR CODE ENDS HERE
            return one_hot

        '''
        Test 1: tf.Tensor([0. 1. 0. 0.], shape=(4,), dtype=float32)
        Test 2: tf.Tensor([0. 0. 1. 0. 0.], shape=(5,), dtype=float32)
        '''
        ```

        ```python
        new_y_test = y_test.map(one_hot_matrix)
        new_y_train = y_train.map(one_hot_matrix)

        print(next(iter(new_y_test)))

        # tf.Tensor([1. 0. 0. 0. 0. 0.], shape=(6,), dtype=float32)

        ```

- Initialize the Parameters
    - initialize a vector of numbers with the Glorot initializer
    - the function to call is tf.keras.initializers.GlorotNormal, which draws samples from a truncated normal distribution centered on 0, with stddev = sqeer(2 / (fan_in + fan_out)), where fan_in is the number of input units and fan_out is the number of output units, both in the weight tensor
    - to initialize with zeros, or ones, use tf.zeros() or tf.ones()
    - Exercise 4 - initialize_parameters
        - implement the function below to take in a shape and to return an array of numbers using the GloroNormal initializer
            - tf.keras.initializers.GlorotNormal(seed=1)
            - tf.Variable(initializer(shape=())

            ```python
            # GRADED FUNCTION: initialize_parameters

            def initialize_parameters():
                """
                Initializes parameters to build a neural network with TensorFlow. The shapes are:
                                    W1 : [25, 12288]
                                    b1 : [25, 1]
                                    W2 : [12, 25]
                                    b2 : [12, 1]
                                    W3 : [6, 12]
                                    b3 : [6, 1]

                Returns:
                parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
                """

                initializer = tf.keras.initializers.GlorotNormal(seed=1)   
                #(approx. 6 lines of code)
                # W1 = ...
                # b1 = ...
                # W2 = ...
                # b2 = ...
                # W3 = ...
                # b3 = ...
                # YOUR CODE STARTS HERE
                W1 = tf.Variable(initializer(shape=(25, 12288)))
                b1 = tf.Variable(initializer(shape=(25, 1)))
                W2 = tf.Variable(initializer(shape=(12, 25)))
                b2 = tf.Variable(initializer(shape=(12, 1)))
                W3 = tf.Variable(initializer(shape=(6, 12)))
                b3 = tf.Variable(initializer(shape=(6, 1)))
                # YOUR CODE ENDS HERE

                parameters = {"W1": W1,
                              "b1": b1,
                              "W2": W2,
                              "b2": b2,
                              "W3": W3,
                              "b3": b3}

                return parameters

            '''
            W1 shape: (25, 12288)
            b1 shape: (25, 1)
            W2 shape: (12, 25)
            b2 shape: (12, 1)
            W3 shape: (6, 12)
            b3 shape: (6, 1)
            '''
            ```

Building Your First NN in TensorFlow

- build a NN using TensorFlow
- implement forward prop
- retrieve the grads and train the model
- Implement Forward Propagation
    - one of TensorFlow’s great strengths lies in the fact that only the forward prop needs to be implemented and it will keep track of the operations to calculate back prop automatically
    - Exercise 5 - forward_propagation
        - implement the forward_propagation function
        - use on the tf api
            - tf.math.add
            - tf.linalg.matmul
            - tf.keras.activations.relu
        - do not apply softmax

        ```python
        # GRADED FUNCTION: forward_propagation

        def forward_propagation(X, parameters):
            """
            Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR

            Arguments:
            X -- input dataset placeholder, of shape (input size, number of examples)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                          the shapes are given in initialize_parameters

            Returns:
            Z3 -- the output of the last LINEAR unit
            """

            # Retrieve the parameters from the dictionary "parameters" 
            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']
            W3 = parameters['W3']
            b3 = parameters['b3']

            #(approx. 5 lines)                   # Numpy Equivalents (NumPy not to be used. Use TF API):
            # Z1 = ...                           # Z1 = np.dot(W1, X) + b1
            # A1 = ...                           # A1 = relu(Z1)
            # Z2 = ...                           # Z2 = np.dot(W2, A1) + b2
            # A2 = ...                           # A2 = relu(Z2)
            # Z3 = ...                           # Z3 = np.dot(W3, A2) + b3
            # YOUR CODE STARTS HERE
            Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
            A1 = tf.keras.activations.relu(Z1)
            Z2 = tf.math.add(tf.linalg.matmul(W2, A1), b2)
            A2 = tf.keras.activations.relu(Z2)
            Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)
            # YOUR CODE ENDS HERE

            return Z3

        '''
        tf.Tensor(
        [[-0.13430887  0.14086473]
         [ 0.21588647 -0.02582335]
         [ 0.7059658   0.6484556 ]
         [-1.1260961  -0.9329492 ]
         [-0.20181894 -0.3382722 ]
         [ 0.9558965   0.94167566]], shape=(6, 2), dtype=float32)
        '''
        ```

- Compute the Total Loss
    - now define the loss function
    - a classification problem with six labels means a categorical cross entropy
    - compute the cost value which sums the losses over the whole batch of samples, then divide the sum by the total number of samples
        - the compute_total_loss function will only take care of summing the losses from one mini-batch of samples
        - as the model is being training, which will call compute_total_loss function once per min-batch, accumulate the sums from each of the mini-batches
        - finish with the division by the total number of samples to get eh final cost value
    - computing the total loss instead of mean loss can make sure the final cost value is consistent
    - if the min-batch size is four but there are just five examples int eh whole batch, then the last mini-batch is going to have one sample only
    - considering the five samples, losses to be [0,1,2,3,4] respectively, the final cost should be their average which is 2
    - adopting the total loss approach will get the same answer
    - however, the mean loss approach will first get 1.5 and 4 for the two mini-batches, and then finally 2.75 after taking average of them, which is different from the desired result of 2
    - therefore, the total loss approach is adopted
    - save the average to the end
    - Exercise 6 - compute_total_loss
        - implement the total loss function
        - use it to compute the total loss of a batch of samples
        - sum the losses across many batches, and divide the sum by the total number of samples to get the cost values
        - note that the y_pred and y_true inputs of tf.keras.losses.categorical_crossentropy are expected to be of shape (number of examples, num_classes)
        - tf.reduce_sum does the summation over the examples
        - tf.keras.losses.categorical_crossentropy will apply softmax by setting parameter from_logits=True
        - [short explanation](https://community.deeplearning.ai/t/week-3-assignment-compute-total-loss-try-to-set-from-logits-false/243049/2?u=paulinpaloalto)

        ```python
        # GRADED FUNCTION: compute_total_loss 

        def compute_total_loss(logits, labels):
            """
            Computes the total loss

            Arguments:
            logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
            labels -- "true" labels vector, same shape as Z3

            Returns:
            total_loss - Tensor of the total loss value
            """

            #(1 line of code)
            # remember to set `from_logits=True`
            # total_loss = ...
            # YOUR CODE STARTS HERE
            total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))   
            # YOUR CODE ENDS HERE
            return total_loss
        ```

    - when using sum of losses for gradient computation, its important to reduce the learning rate as the size of the mini-batch increases
    - this ensures that too large of steps are taken towards the minimum
- Train the model
    - specify the type of optimizer in one line, with tf.keras.optimizers.Adam, and then call it within the trianing loop
    - notice the tape.gradient function: this allows to retrieve the operations recorded for automatic differentiation inside the GradientTape block
    - then, calling the optimizer method apply_gradients, will apply the optimizer’s update rules to each training parameter
    - take an important extra step that’s been added tot he batch training process: tf.Data.dataset = dataset.prefetch(8)
    - what this does is prevent a memory bottleneck that can occur when reading from disk
    - prefetch() sets aside some data and keeps it ready for when it’s needed
    - it does this by creating a source dataset from the input data, applying a transformation to preprocess the data, then iterating over the dataset the specified number of elements at a time
    - works because the iteration is streaming, so the data doesn’t need to fit into the memory

        ```python
        def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
                  num_epochs = 1500, minibatch_size = 32, print_cost = True):
            """
            Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

            Arguments:
            X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
            Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
            X_test -- training set, of shape (input size = 12288, number of training examples = 120)
            Y_test -- test set, of shape (output size = 6, number of test examples = 120)
            learning_rate -- learning rate of the optimization
            num_epochs -- number of epochs of the optimization loop
            minibatch_size -- size of a minibatch
            print_cost -- True to print the cost every 10 epochs

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

            costs = []                                        # To keep track of the cost
            train_acc = []
            test_acc = []

            # Initialize your parameters
            #(1 line)
            parameters = initialize_parameters()

            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']
            W3 = parameters['W3']
            b3 = parameters['b3']

            optimizer = tf.keras.optimizers.Adam(learning_rate)

            # The CategoricalAccuracy will track the accuracy for this multiclass problem
            test_accuracy = tf.keras.metrics.CategoricalAccuracy()
            train_accuracy = tf.keras.metrics.CategoricalAccuracy()

            dataset = tf.data.Dataset.zip((X_train, Y_train))
            test_dataset = tf.data.Dataset.zip((X_test, Y_test))

            # We can get the number of elements of a dataset using the cardinality method
            m = dataset.cardinality().numpy()

            minibatches = dataset.batch(minibatch_size).prefetch(8)
            test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
            #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
            #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster 

            # Do the training loop
            for epoch in range(num_epochs):

                epoch_total_loss = 0.

                #We need to reset object to start measuring from 0 the accuracy each epoch
                train_accuracy.reset_states()

                for (minibatch_X, minibatch_Y) in minibatches:

                    with tf.GradientTape() as tape:
                        # 1. predict
                        Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

                        # 2. loss
                        minibatch_total_loss = compute_total_loss(Z3, tf.transpose(minibatch_Y))

                    # We accumulate the accuracy of all the batches
                    train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))

                    trainable_variables = [W1, b1, W2, b2, W3, b3]
                    grads = tape.gradient(minibatch_total_loss, trainable_variables)
                    optimizer.apply_gradients(zip(grads, trainable_variables))
                    epoch_total_loss += minibatch_total_loss

                # We divide the epoch total loss over the number of samples
                epoch_total_loss /= m

                # Print the cost every 10 epochs
                if print_cost == True and epoch % 10 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
                    print("Train accuracy:", train_accuracy.result())

                    # We evaluate the test set every 10 epochs to avoid computational overhead
                    for (minibatch_X, minibatch_Y) in test_minibatches:
                        Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                        test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
                    print("Test_accuracy:", test_accuracy.result())

                    costs.append(epoch_total_loss)
                    train_acc.append(train_accuracy.result())
                    test_acc.append(test_accuracy.result())
                    test_accuracy.reset_states()

            return parameters, costs, train_acc, test_acc
        ```

        ```python
        parameters, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, num_epochs=100)

        '''
        Cost after epoch 0: 1.830244
        Train accuracy: tf.Tensor(0.17037037, shape=(), dtype=float32)
        Test_accuracy: tf.Tensor(0.2, shape=(), dtype=float32)
        ...
        Cost after epoch 90: 0.744699
        Train accuracy: tf.Tensor(0.7546296, shape=(), dtype=float32)
        Test_accuracy: tf.Tensor(0.69166666, shape=(), dtype=float32)
        '''
        ```

        ```python
        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(0.0001))
        plt.show()
        ```

        ```python
        # Plot the train accuracy
        plt.plot(np.squeeze(train_acc))
        plt.ylabel('Train Accuracy')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(0.0001))
        # Plot the test accuracy
        plt.plot(np.squeeze(test_acc))
        plt.ylabel('Test Accuracy')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(0.0001))
        plt.show()

        ```