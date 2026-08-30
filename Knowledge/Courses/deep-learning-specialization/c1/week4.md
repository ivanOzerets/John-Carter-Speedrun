Deep L-layer NN

- What is a deep neural network
    - more hidden layers

    - shallow means less layers
    - number of layers can be a hyperparameter
- Deep NN notation
    - L is the number of layers
    - n[l] = number of units in layer l

    - n[0] = nx = 3
    - n[L] = n[4] = 1
    - a[l] = activations in layer l
    - a[l] = g[l](z[l])
    - input features x = a[0]
    - a[l] = y_hat

Forward Propagation in a Deep Network

- for a single training example

- x is a[0]
- general forward prop equation
    - Z[l] = W[l]a[l-1] + b[l]
    - a[l] = g[l](Z[l])
- vectorized

- looks like there is a for loop on the layers → okay

Getting your Matrix Dimensions Right

- Parameters W[l] and b[l]
    - number the units in the layer
    - what does W need to be
    - W[l] = (n[l], n[l-1])
    - b needs to be the same dimension as W x a to add
    - b[l] = (n[l], 1)
    - dW and db should be the same
- Vectorized Implementation
    - dimensions of W stay the same because they don’t change with each training example
    - through broadcasting, b becomes (n[l], m) dimensions
    - Z[l] and A[l]: (n[l], m)
    - dZ and dA are the same

Why Deep Representations?

- Intuition about deep representation
    - first layer could be feature detector
    - second layer could be grouping edges to detect parts of faces
    - third layer could group faces into categories
    - convolutional nets → look at small to large areas
    - with audio first level can be low level wave form features → units of sounds → words → sentence
- Circuit theory and deep learning
    - informally: there are functions you can compute with a small L-layer dep NN that shallower networks require exponentially more hidden units to compute
    - if you try to compute the xor of all the input features
    - with circuits the depth will be of order O(logn)
    - without multiple hidden layers, a NN will need to be exponentially large to compute the possible 2^n configurations of the inputs

Building Blocks of Deep NN

- Forward and backward functions
    - forward pass: input a[l-1], output a[l]

    - cache the value Z[l]
    - backward pass: input da[l] and Z[l], output da[l-1] and dW[l], db[l]

    - with many hidden layers

    - conceptually, the cache is storing the value of Z for backward functions
    - the cache is also a good way to get the W and b into the backward function

Forward and Backward Propagation

- Forward propagation for layer l
    - input a[l-1]
    - output a[l], cache (z[l], W[l], b[l])
    - Z[l] = W[l] x A[l-1] + b[l]
    - A[l] = g[l](Z[l])
- Backward propagation for layer l
    - input da[l]
    - output da[l-1], dW[l], db[l]
    - you also need a[l-1] in the cache

- Summary
    - the input to back prop is the derivative of the loss function with respect to y_hat

Optional Reading: Feedforward NN in Depth

- [Feedforward NN in Depth, Part 1: Forward and Backward Prop](https://jonaslalin.com/2021/12/10/feedforward-neural-networks-part-1/)
- [Feedforward NN in Depth, Part 2: Activation Functions](https://jonaslalin.com/2021/12/21/feedforward-neural-networks-part-2/)
- [Feedforward NN in Depth, Part 3: Cost Functions](https://jonaslalin.com/2021/12/22/feedforward-neural-networks-part-3/)

Parameters vs Hyperparameters

- What are hyperparameters?
    - parameters: W[1], b[1], …
    - hyperparameters
        - learning rate alpha
        - number of iterations
        - number of hidden layers L
        - number of hidden units
        - choice of activation function
- Applied deep learning is a very empirical process
    - idea, code, experiment
    - difficult to know best values for hyperparameters
    - intuitions on hyperparameters some times doesn’t transfer from on application to another
    - best values might change over time

What does this have to do with the brain?

- not a whole lot
- it’s like the brain is oversimplifying
- scientist now don’t know what a single neuron is doing

- deep learning is being very good at learning very flexible function, very complex functions, learning mappings

References

- [Stanford CS231n Deep Learning for Computer Vision Course](https://cs231n.github.io/)