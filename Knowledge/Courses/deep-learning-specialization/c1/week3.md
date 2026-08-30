Neural Networks Overview

- both stacks of rows, layer, have a z calculation and an a calculation

- new notation is z^[1], square bracket refer to layers, round bracket refer to the ith training example

Neural Network Representation

- input feature are called the input layer
- second layer is the hidden layer, ‘hidden’ because you can’t see the values explicitly
- last layer is called the output layer
- a^[0] = X
- a stands for activations, and refers to the values that different layers of the NN are passing on to the subsequent layers
- a^[1] = [a^[1], … , a^[n]]

- input layer does not count, hidden layer is layer 1
- 2 layer NN ^^^
- hidden layer has w^[1] and b^[1]

Computing a NN’s Output

- the circle, or a neuron, represents two steps of computation
- computes z and the activation

- performing the z and a computations in a loops in inefficient

- dot product the weights and the input features
- the weights for a whole layer are represented with a capital letter  W
- activation of a[1] is sigmoid of z[1]
- a[0] is an alias for X

Vectorizing Across Multiple Examples

- a[2] → layer 2, example i
- remove the computation of each training example through the network with vectorization
- A[1] is the list off all activations from a layer

Explanation for Vectorized Implementation

- Justification for vectorized implementation
    - dot product creates a column for z and adding z’s together creates the final Z for the layer

    - broadcast just adds b to the end of the column
    - layers are roughly doing the same thing

Activation Functions

- options other than sigmoid
- g(z) as the general function
- tanh(z) almost always works better

- just the sigmoid function but shifted
- better because the mean of the activation is closer to zero
- has the effect of centering the data
- sigmoid is almost never used, except for the output layer
- the downside of both is how the gradient becomes very small if the values are either very large or very small → slow down GD
- ReLU → a = max(0,z)
- leaky ReLU → max(0.01z, z)

- advantage is that most derivatives are different from zero
- less of this effect of the slope of the function going to zero
- in practice, most zs will be above zero
- test multiple activation functions

Why do you need Non-Linear Activation Functions?

- g(z) → linear activation, linear
- just computing y as a linear function of the input features
- the network will just be on neuron
- linear on the hidden and sigmoid on the end is not more expressive than the standard logistic regression alg
- composition of two linear function is another linear function
- output linear for something like housing prices

Derivatives of Activation Functions

- Sigmoid activation function
    - slope of g(x) at z

- Tang activation function
    - slope of g(z) at z = 1 - (tang(z))^2
    - g(z) = z = e^z - e^-z / e^z + e^-z

- ReLU and Leaky ReLU activation function
    - g(z) = max(0,1)

    - g(z) = max(0.01z, z)

Gradient Descent for NN

- parameters: W[1], b[1], W[2]. b[2], n[0] input feature, n[1] hidden, n[2] = 1
- cost function: J(W[1], b[1], W[2]. b[2]) = 1/m sum loss(y_hat, y)
- to train, perform GD
- initialize parameters randomly
- repeat
    - compute predictions (y^i, i=1..m)
    - dW[1], db[1], …
    - W[1] = W[1] - alpha dW[1]
    - b[1] = b[1] - alpha db[1]
    - …
- Formulas for computing derivatives
    - forward prop
        - Z[1] = W[1]X + b[1]
        - A[1] = g[1](Z[1])
        - Z[2] = W[2]A[1] + b[2]
        - A[2] = g[2](Z[2])
    - backprop
        - dZ[2] = A[2] - Y
        - dW[2] = 1/m dZ[2]A[1].T
        - db[2] = 1/m np.sum(dZ[2], axis, 1, keepdims = True) (keepdims prevents rank one arrays)
        - dZ[1] = W[2].T dz[2] x g[z]’(Z[1])
        - dW[1] = q/m dZ[1]X.T
        - db[1] = 1/m np.sum(dZ[1], axis = 1, keepdims = True)

Backpropagation Intuition

- Computing gradients
    - Logistic regression

- NN gradients
    - computes da[2], dz[2], dW[2], db[2], da[1], dz[1], W[1], b[1]
    - input x for supervised learning is fixed so no need to take derivative of the input
    - computation of da and dz are collapsed into one step

- Summary of GD
    - vectorized implementation, previous was for a single training example

Random Initialization

- for logistic regression, initializing all weights to zero works, but doesn’t for NN
- What happens if you initialize weights to zero
    - biased terms to zero is okay
    - the activation for the neurons in a given layer will be the same
    - in backprop, both derivatives will be the same
    - the neurons will be symmetric and will be the same neuron
    - both hidden units will compute the same function
- W[1] = np.random.randn((2,2)) x 0.01
- b[1] = np.zeros((2,1))
- same for W[2] and b[2]
- b does not have the ‘symmetry breaking problem’
- small values are preferred because activation values can be sensitive to big or small values
- training deep NN, different constants than 0.01 can be better

Ian Goodfellow Interview

- GANs - generative adversarial networks → producing more training data that is imaginary
- sparse coding
- semi-supervised learning, generating training data, and even simulating scientific experiments
- Deep Learning textbook
- write good code and put it on github
- network level security vs application level security