Notation

- superscript[l] denotes an object of the lth layer
- superscript (i) denotes an object from the ith example
- subscript i denotes the ith entry of the vector
- nh, nw and nc denote respectably the height, width and number of channels of a given layer
- to reference a specific layer l, write nh[l].. nw[l]. nc[l]
- nh_prev, nw_prev, nc_prev denote respectively the height, width, and number of channels of the previous layer
- to reference a specific layer l, denoted with nh[l-1], nw[l-1], nc[l-1]

Packages

```python
import numpy as np
import h5py
import matplotlib.pyplot as plt
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
```

Outline of the Assignment

- implementing the building blocks of a convolutional NN
- convolution function
    - zero padding
    - convolve window
    - convolution forward
    - convolution backward
- pooling functions
    - pooling forward
    - create mask
    - distribute value
    - pooling backward
- from scratch, there are tf equivalents

- for ever forward function, there is a corresponding backward equivalent
- hence, at every step of the forward module, store some parameters in a cache
- the parameters are used to compute gradients during back prop

Convolutional NN

- although programming frameworks make convolutions easy to use, they remain one of the ardest concepts to understandin Deep learning
- a convolution layer transforms an input volume into an output volume of different size

- Zero-Padding
    - zero-padding adds zeros around the border of an image

    - the main benefits of padding are
        - it allows the use of a CONV layer without necessarily shrinking the height and width of the volumes
        - this is important for building deeper networks, since otherwise the height/width would shrink with deeper layers
        - an important special case is the same convolution, in which the height/width is exactly preserved after on layer
        - it helps to keep more of the information at the boarder of an image
        - without padding, very few values at the next layers would be affected by pixels at the edges of an image
    - Exercise 1 - zero_pad
        - implement the following function, which pads all the images of a batch of examples X with zeros
        - use np.pad
        - to pad the array a of shape (5,5,5,5,5) with pad=1 for the 2nd dimensions, pad=3 for the 4th dimension and pad=0 for the rest

            ```python
            a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), mode='constant', constant_values = (0,0))
            ```

        ```python
        # GRADED FUNCTION: zero_pad

        def zero_pad(X, pad):
            """
            Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
            as illustrated in Figure 1.

            Argument:
            X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
            pad -- integer, amount of padding around each image on vertical and horizontal dimensions

            Returns:
            X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
            """

            #(≈ 1 line)
            # X_pad = None
            # YOUR CODE STARTS HERE
            X_pad = np.pad(X, ((0,0), (pad, pad), (pad, pad), (0,0)), mode='constant', constant_values = (0,0))    
            # YOUR CODE ENDS HERE

            return X_pad
        ```

- Single Step Convolution
    - implement a single step of convolution, in which a filter is applied to a single position of the input
    - this will be used to build a convolutional unit
        - takes an input volume
        - applies a filter at every position of the input
        - outputs another volume (usually of a different size)

    - in compute vision applications, each value in the matrix on the left corresponds to a single pixel value
    - convolve a 3x3 filter with the image by multiplying its values element0wise with the original matrix, then summing them up and adding a bias
    - implement a single step of convolution, corresponding to applying a filter to just one of the positions to get a single real-valued output
    - Exercise 2 - conv_single_step
        - the variable b will be passed in as a numpy array
        - adding a scaler to a numpy array, results in a numpy array
        - in the special case of a numpy array contain a single value, cast it as a float to convert to a scalar

        ```python
        # GRADED FUNCTION: conv_single_step

        def conv_single_step(a_slice_prev, W, b):
            """
            Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
            of the previous layer.

            Arguments:
            a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
            W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
            b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

            Returns:
            Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
            """

            #(≈ 3 lines of code)
            # Element-wise product between a_slice_prev and W. Do not add the bias yet.
            # s = None
            # Sum over all entries of the volume s.
            # Z = None
            # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
            # Z = None
            # YOUR CODE STARTS HERE
            s = np.multiply(a_slice_prev, W)
            Z = np.sum(s)
            Z += float(b)
            # YOUR CODE ENDS HERE

            return Z

        # Z = -6.999089450680221
        ```

- Convolutional NN - Forward Pass
    - take many filters and convolve them on the input
    - each ‘convolution’ gives a 2D matrix output
    - then stack there outputs to get a 3D volume

    - Exercise 3 - conv_forward
        - implement the function below to convolve the filters W on an input activation A_prev
        - this function takes the following inputs
            - A_prev, the activations output by the previous layer (for a batch of m inputs)
            - weights are denoted by W the filter window size of f by f
            - the bias vector is b, where each filter has its own (single) bias
        - to select a 2x2 slice at the upper left corner of a matrix a_prev (shape (5,5,3,))

            ```python
            a_slice_prev = a_prev[0:2,0:2,:]
            ```

        - notice how this gives a 3D slice that has height 2, width 2, and depth 3
        - depth is the number of channels
        - this will be useful when you will define a_slice_prev below, sing the start/end indexes to be defined
        - to define a_slice, first define its corners vert_start, vert_end, horiz_start and horiz_end

        - the formulas relating the output shape of the convolution to the input shape are

        - without vectorization for now
        - use array slicing for the following variables: a_prev_pad, W, b
        - to decide how to get the vert_start, vert_end, horiz_start, horiz_end, remember that these are indices of the previous layer
            - the output layer’s indices are denoted by h and w
        - make sure that a_slice_prev has a height, width and depth
        - remember that a_prev_pad is a subset of A_prev_pad

        ```python
        # GRADED FUNCTION: conv_forward

        def conv_forward(A_prev, W, b, hparameters):
            """
            Implements the forward propagation for a convolution function

            Arguments:
            A_prev -- output activations of the previous layer, 
                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
            W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
            b -- Biases, numpy array of shape (1, 1, 1, n_C)
            hparameters -- python dictionary containing "stride" and "pad"

            Returns:
            Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
            cache -- cache of values needed for the conv_backward() function
            """

            # Retrieve dimensions from A_prev's shape (≈1 line)  
            # (m, n_H_prev, n_W_prev, n_C_prev) = None

            # Retrieve dimensions from W's shape (≈1 line)
            # (f, f, n_C_prev, n_C) = None

            # Retrieve information from "hparameters" (≈2 lines)
            # stride = None
            # pad = None   

            # Compute the dimensions of the CONV output volume using the formula given above. 
            # Hint: use int() to apply the 'floor' operation. (≈2 lines)
            # n_H = None
            # n_W = None

            # Initialize the output volume Z with zeros. (≈1 line)
            # Z = None

            # Create A_prev_pad by padding A_prev
            # A_prev_pad = None

            # for i in range(None):               # loop over the batch of training examples
                # a_prev_pad = None               # Select ith training example's padded activation
                # for h in range(None):           # loop over vertical axis of the output volume
                    # Find the vertical start and end of the current "slice" (≈2 lines)
                    # vert_start = None
                    # vert_end = None

                    # for w in range(None):       # loop over horizontal axis of the output volume
                        # Find the horizontal start and end of the current "slice" (≈2 lines)
                        # horiz_start = None
                        # horiz_end = None

                        # for c in range(None):   # loop over channels (= #filters) of the output volume

                            # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                            # a_slice_prev = None

                            # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                            # weights = None
                            # biases = None
                            # Z[i, h, w, c] = None
            # YOUR CODE STARTS HERE
            (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
            (f, f, n_C_prev, n_C) = W.shape

            stride = hparameters["stride"]
            pad = hparameters["pad"]

            n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
            n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

            Z = np.zeros((m, n_H, n_W, n_C))

            A_prev_pad = zero_pad(A_prev, pad)

            for i in range(m):
                a_prev_pad = A_prev_pad[i,:,:,:]

                for h in range(n_H):
                    vert_start = h * stride
                    vert_end = h * stride + f

                    for w in range(n_W):
                        horiz_start = w * stride
                        horiz_end = w * stride + f

                        for c in range(n_C):
                            a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                            weights = W[:,:,:,c]
                            biases = b[:,:,:,c]
                            Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
            # YOUR CODE ENDS HERE

            # Save information in "cache" for the backprop
            cache = (A_prev, W, b, hparameters)

            return Z, cache

        '''
        Z's mean =
         0.5511276474566768
        Z[0,2,1] =
         [-2.17796037  8.07171329 -0.5772704   3.36286738  4.48113645 -2.89198428
         10.99288867  3.03171932]
        cache_conv[0][1][2][3] =
         [-1.1191154   1.9560789  -0.3264995  -1.34267579]
        '''
        ```

    - finally, a CONV layer should also contain an activation, in which case add the following

        ```python
        # Convolve the window to get back one output neuron
        Z[i, h, w, c] = ...
        # Apply activation
        A[i, h, w, c] = activation(Z[i, h, w, c])
        ```

Pooling Layer

- the pooling (POOL) layer reduces the height and width of the input
- help reduce computation, as well as helps make feature detectors more invariant to its position in the input
- two types of pooling
    - max-pooling layer: slides an (f, f) window over the input and stores the max value of the window in the output
    - average-pooling layer: slides an (f, f) window over the input and stores the average value of the window in the output

- these pooling layers have no parameters for backprop to train
- however, they have hypers such as the window size f
- this specifies the height and width of the f x f window used to compute a max or average over
- Forward Pooling
    - implement MAX-POOL and AVG-POOL
    - Exercise 4 - pool-forward
        - implement the forward pass of the pooling layer
        - as there’s no padding, the formulas binding the output shape of the pooling to the input shape is

    ```python
    # GRADED FUNCTION: pool_forward

    def pool_forward(A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """

        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]

        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              

        # for i in range(None):                         # loop over the training examples
            # for h in range(None):                     # loop on the vertical axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                # vert_start = None
                # vert_end = None

                # for w in range(None):                 # loop on the horizontal axis of the output volume
                    # Find the vertical start and end of the current "slice" (≈2 lines)
                    # horiz_start = None
                    # horiz_end = None

                    # for c in range (None):            # loop over the channels of the output volume

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        # a_prev_slice = None

                        # Compute the pooling operation on the slice. 
                        # Use an if statement to differentiate the modes. 
                        # Use np.max and np.mean.
                        # if mode == "max":
                            # A[i, h, w, c] = None
                        # elif mode == "average":
                            # A[i, h, w, c] = None

        # YOUR CODE STARTS HERE
        for i in range(m):
            for h in range(n_H):
                vert_start = h * stride
                vert_end = h * stride + f

                for w in range(n_W):
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    for c in range(n_C):
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        # YOUR CODE ENDS HERE

        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)

        # Making sure your output shape is correct
        #assert(A.shape == (m, n_H, n_W, n_C))

        return A, cache

    '''
    mode = max
    A.shape = (2, 2, 2, 3)
    A[0] =
     [[[1.74481176 0.90159072 1.65980218]
      [1.74481176 1.6924546  1.65980218]]

     [[1.13162939 1.51981682 2.18557541]
      [1.13162939 1.6924546  2.18557541]]]

    mode = average
    A.shape = (2, 2, 2, 3)
    A[1] =
     [[[-0.17313416  0.32377198 -0.34317572]
      [ 0.02030094  0.14141479 -0.01231585]]

     [[ 0.42944926  0.08446996 -0.27290905]
      [ 0.15077452  0.28911175  0.00123239]]]
    '''
    ```

- the 2D output of the convolution is called the feature map

Backpropagation in Convolutional NN

- in modern deep learning frameworks, only the forward pass needs to be implemented
- the framework takes care of the backward pass
- most deep learning engineers don’t need to bother with the details of the backward pass
- the backward pass for convolutional networks is complicated
- in convolutional NN, derivatives are calculated w.r.t. the cost in order to update the parameters
- the backprop equations are not trivial
- Convolutional Layer Backward Pass
    - implement the backward pass for a CONV layer
    - Computing dA
        - this is the formula for computing dA w.r.t the cost for a certain filter Wc and a given training example

        - where Wc is a filter and dZhw is a scalar corresponding to the gradient of the cost w.r.t the output of the conv layer Z at the hth row and wth column (corresponding to the dot product taken at the ith stride left and jth stride down
        - note that at each time, multiply the same filter Wc by a different dZ when updating dA
        - done mainly because when computing the forward propagation, each filter is dotted and dammed by a different a_slice
        - therefor, when computing the backprop for dA, just add the gradients of al the a_slices
        - in code, inside the appropriate for-loops, this formula translates to

            ```python
            da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
            ```

    - Computing dW
        - this is the formula for computing dWc (dWc is the derivative of one filter) w.r.t the loss

        - where a_slice corresponds to the slice which was used to generate the activation Zij
        - hence, this ends up giving the gradient for W w.r.t that slice
        - since it is the same W, add up all such gradients to get dW

            ```python
            dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
            ```

    - Computing db
        - this is the formula for computing db w.r.t the cost for a certain filter Wc

        - db is computed by summing dZ
        - in this case, just summing over all the gradients of the conv output Z w.r.t the cost

            ```python
            db[:,:,:,c] += dZ[i, h, w, c]
            ```

- Exercise 5 - conv_backward
    - implement the conv_backward function
    - sum over all training examples, filters, heights, and widths

    ```python
    def conv_backward(dZ, cache):
        """
        Implement the backward propagation for a convolution function

        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()

        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
              numpy array of shape (1, 1, 1, n_C)
        """    

        # Retrieve information from "cache"
        # (A_prev, W, b, hparameters) = None
        # Retrieve dimensions from A_prev's shape
        # (m, n_H_prev, n_W_prev, n_C_prev) = None
        # Retrieve dimensions from W's shape
        # (f, f, n_C_prev, n_C) = None

        # Retrieve information from "hparameters"
        # stride = None
        # pad = None

        # Retrieve dimensions from dZ's shape
        # (m, n_H, n_W, n_C) = None

        # Initialize dA_prev, dW, db with the correct shapes
        # dA_prev = None                          
        # dW = None
        # db = None

        # Pad A_prev and dA_prev
        # A_prev_pad = zero_pad(A_prev, pad)
        # dA_prev_pad = zero_pad(dA_prev, pad)

        #for i in range(m):                       # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            # a_prev_pad = None
            # da_prev_pad = None

            #for h in range(n_H):                   # loop over vertical axis of the output volume
            #    for w in range(n_W):               # loop over horizontal axis of the output volume
            #        for c in range(n_C):           # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        # vert_start = None
                        # vert_end = None
                        # horiz_start = None
                        # horiz_end = None

                        # Use the corners to define the slice from a_prev_pad
                        # a_slice = None

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        # da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += None
                        # dW[:,:,:,c] += None
                        # db[:,:,:,c] += None

            # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            # dA_prev[i, :, :, :] = None
        # YOUR CODE STARTS HERE
        (A_prev, W, b, hparameters) = cache
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape

        stride = hparameters["stride"]
        pad = hparameters["pad"]

        (m, n_H, n_W, n_C) = dZ.shape

        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)

        for i in range(m):
            a_prev_pad = A_prev_pad[i,:,:,:]
            da_prev_pad = dA_prev_pad[i,:,:,:]

            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * stride
                        vert_end = h * stride + f
                        horiz_start = w * stride
                        horiz_end = w * stride + f

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]

            dA_prev[i,:,:,:] = da_prev_pad[pad:-pad, pad:-pad, :]
        # YOUR CODE ENDS HERE

        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev, dW, db

    '''
    dA_mean = 1.4524377775388075
    dW_mean = 1.7269914583139097
    db_mean = 7.839232564616838
    '''
    ```

Pooling Layer - Backward Pass

- implement the backward pass for the pooling layer, starting with the MAX-POOL layer
- even though a pooling layer has no parameters for backprop to update, still need to backpropagate the gradient through the pooling layer in order to compute gradients for layers that came before the pooling layer
- Max Pooling - Backward Pass
    - build a helper function called create_mask_from_window() which does

    - this function creates a mask matrix which keeps track of where the maximum of the matrix is
    - true indicates the position of the maximum in X, the other entries are false
    - Exercise 6 - creat_mask_from_window
        - implement creat-mask_from_window()
        - helpful for pooling backward
        - np.max() computes the maximum of an array
        - a matrix X and a scalar x: A = (X == x) will return a matrix A of the same size as X such that
            - A[i,j] = True if X[i,j] = x
            - A[i,j] = False if X[i,j] ≠ x
        - don’t need to consider cases where there are several maxima in a matrix

        ```python
        def create_mask_from_window(x):
            """
            Creates a mask from an input matrix x, to identify the max entry of x.

            Arguments:
            x -- Array of shape (f, f)

            Returns:
            mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
            """    
            # (≈1 line)
            # mask = None
            # YOUR CODE STARTS HERE
            mask = (x == np.max(x))
            # YOUR CODE ENDS HERE
            return mask
        ```

    - why keep track of the position of the max
    - because this is the input value that ultimately influenced the output, and therefore the cost
    - backprop is computing gradients w.r.t the cost, so anything that influences the ultimate cost should have a non-zero gradient
    - so, backprop will propagate the gradient back tot his particular input value that had influenced the cost
- Average Pooling - Backward Pass
    - in max pooling, for each input window, all the influence on the output came from a single input value → the max
    - in average pooling, ever element of the input window has equal influence
    - so to implement backprop, now implement a helper function that reflects this
    - for example, if average pooling on the forward pass using a 2x2 filter, then the mask for the backward pass would be

    - this implies that each position in the dZ matrix contributes equally to output because in the forward pass, an average was taken
    - Exercise 7 - distribute_value
        - implement the function to equally distribute a value dz through a matrix of dimension shape

        ```python
        def distribute_value(dz, shape):
            """
            Distributes the input value in the matrix of dimension shape

            Arguments:
            dz -- input scalar
            shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

            Returns:
            a -- Array of size (n_H, n_W) for which we distributed the value of dz
            """    
            # Retrieve dimensions from shape (≈1 line)
            # (n_H, n_W) = None

            # Compute the value to distribute on the matrix (≈1 line)
            # average = None

            # Create a matrix where every entry is the "average" value (≈1 line)
            # a = None
            # YOUR CODE STARTS HERE
            (n_H, n_W) = shape

            average = dz / (n_H * n_W)

            a = np.ones((n_H, n_W)) * average
            # YOUR CODE ENDS HERE
            return a
        ```

- Putting it Together: Pooling Backward
- compute backprop on a pooling layer
    - Exercise 8 - pool_backward
        - implement the pool_backward function in both modes (max and average)
        - once again, use 4 for-loops
        - if it is equal to ‘average’ → distribute_value() function
        - otherwise, the mode is equal to ‘max’, create a mask with creat_mask_from_window() and multiply it by the corresponding value of dA

        ```python
        def pool_backward(dA, cache, mode = "max"):
            """
            Implements the backward pass of the pooling layer

            Arguments:
            dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
            cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
            mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

            Returns:
            dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
            """
            # Retrieve information from cache (≈1 line)
            # (A_prev, hparameters) = None

            # Retrieve hyperparameters from "hparameters" (≈2 lines)
            # stride = None
            # f = None

            # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
            # m, n_H_prev, n_W_prev, n_C_prev = None
            # m, n_H, n_W, n_C = None

            # Initialize dA_prev with zeros (≈1 line)
            # dA_prev = None

            # for i in range(None): # loop over the training examples

                # select training example from A_prev (≈1 line)
                # a_prev = None

                # for h in range(n_H):                   # loop on the vertical axis
                    # for w in range(n_W):               # loop on the horizontal axis
                        # for c in range(n_C):           # loop over the channels (depth)

                            # Find the corners of the current "slice" (≈4 lines)
                            # vert_start = None
                            # vert_end = None
                            # horiz_start = None
                            # horiz_end = None

                            # Compute the backward propagation in both modes.
                            # if mode == "max":

                                # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                                # a_prev_slice = None

                                # Create the mask from a_prev_slice (≈1 line)
                                # mask = None

                                # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                                # dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None

                            # elif mode == "average":

                                # Get the value da from dA (≈1 line)
                                # da = None

                                # Define the shape of the filter as fxf (≈1 line)
                                # shape = None

                                # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                                # dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += None
            # YOUR CODE STARTS HERE
            (A_prev, hparamters) = cache

            stride = hparameters["stride"]
            f = hparameters["f"]

            (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
            (m, n_H, n_W, n_C) = dA.shape

            dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

            for i in range(m):
                a_prev = A_prev[i,:,:,:]

                for h in range(n_H):
                    for w in range(n_W):
                        for c in range(n_C):
                            vert_start = h * stride
                            vert_end = h * stride + f
                            horiz_start = w * stride
                            horiz_end = w * stride + f

                            if mode == "max":
                                a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                                mask = create_mask_from_window(a_prev_slice)

                                dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i,h,w,c]

                            elif mode == 'average':
                                da = dA[i,h,w,c]

                                shape = (f, f)

                                dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
            # YOUR CODE ENDS HERE

            # Making sure your output shape is correct
            assert(dA_prev.shape == A_prev.shape)

            return dA_prev
        ```