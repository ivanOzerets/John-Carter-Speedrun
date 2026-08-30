Tuning Process

- Hyperparameters
    - how to optimize hyperparameter tuning process
    - alpha, beta, beta1, beta2, epsilon, layers, hidden units, learning rate decay, mini-batch size
    - alpha is the most important, then beta ~ 0.9, mini-batch-size and hidden units
    - third in importance, learning rate decay, number of layers and adam hypers are static usually
- Try Random Values: don’t use a grid
    - common practice before was to sample from a grid, okay for relatively small

    - random works best, difficult which hypers are more important than others

    - if hyperparameter 1 is very important and hyper 2 is not, grid style search only tests on a small subset of hyper 1 parameters, whereas at random, way more are tried
- Coarse to fine
    - sampling scheme
    - zoom in to a smaller region

Using an Appropriate Scale to pick hyperparameters

- not uniformly, important to pick correct scale
- Picking hypers at random
    - for unit number or number of layers, uniformly at random is okay
    - not true for all hypers
- Appropriate scale for hypers
    - alpha could be as low as 0.001, …, 1
    - random sample would have most samples be between 0.1 and 1
    - search on log scale

    - r = -4 * np.random.rand()
    - alpha = 10^r
    - more generally
        - take the lower bound, a = log_10(lower bound)
        - take the upper bound, b = log_10(upper bound)
        - r sample between [a, b]
        - alpha = 10^r
- Hypers for exponentially weighted averages
    - for momentum hyper beta
    - 0.9 → last 10 values
    - 0.999 → last 1000 values
    - explore between 1-beta
    - same as above, with upper and lower bound, but goes from larger to smaller
    - then beta = 1 - 10^r

    - bad to sample on linear scale because the scale is sensitive the closer to one it gets
    - 0.9000 → 0.9005 not as much impact as 0.999 → 0.9995
    - goes from 10 to 10 as opposed to 1000 to 2000

Hyperparameters Tuning in Practice: Pandas vs Caviar

- how to organize hypers search process
- Re-test hypers occasionally
    - cross-fertilization across different application does occur
    - intuitions however, do get stale
    - data gradually change or serves get upgraded, so retest hypers
    - once every several months
- Babysitting one model vs Training many models in Parallel
    - if you don’t have enough computational recourses for big data
    - see cost function decrease, change learning rate, change other parameters..
    - panada approach → all the eggs in one basket

    - or many models that have different hypers
    - caviar → a lot of eggs

Normalizing Activations in a Network

- created by Sergey Ioffe and Christian Szegedy
- makes hyper search problem easier and makes network more robust
- Normalizing inputs to speed up learning
    - turns contours into something more round and easier for GD to optimize
    - in a deeper model, normalize the activations of the previous layer
    - actually Z before activation is done more often
- Implementing Batch Norm
    - Given some intermediate values in NN z^1, …, z^m
    - take the mean and variance
    - then normalize
    - add epsilon for numerical stability
    - might want a different distribution where you add gamma and beta, learnable parameters of the model
    - if gamma and beta are to be inversed, then z~ would be z
    - now use z~ instead of z
    - applies hidden unit values to be normalized, however, you don’t want to force the hidden unit values to be normalized to mean 0 or standard variance

Fitting Batch Norm into a NN

- Adding Batch Norm to a network
    - apply batch norm (BN), with beta[l] and gamma[l], then feed into the activation g(z~)
    - per layer BN parameters
    - different beta from momentum and weighted averages

    - update beta and gamma with GD (adam)
- Working with mini-batches
    - BN is usually applied with mini-batches
    - parameters: W[l], b[l], beta[l], gamma[l]
    - z[l] = W[l]a[l-1] + b[l]
    - b[l] can be omitted because adding a constant to a matrix that will be scaled by the mean
    - the constant gest cancelled out by the mean subtraction set

    - dimension of beta and gamma are used to scale the mean and variance of each of the hidden units
    - the shapes of beta and gamma are (n[l], 1)
- Implementing GD
    - BN replaced z[l] with z~ with BN
    - backprop, no bias gradient
    - updates params

Why does Batch Norm work

- normalizing inputs takes them on a similar range of values which can speed up learning
- doing a similar thing for hidden layer units
- Learning on shifting input distribution
    - makes weights deeper in the network more robust to changes to weights in earlier layers
    - the distribution may change when new data is provided → covariate shift
    - if a x to y mapping is learning, and the distribution of x changes, the model needs to b e retrained

- Why this is a problem with NN
    - from the third layer, for example, the nodes have learned some W and b
    - from its perspective it gets some values a
    - its job is to map to yhat
    - from the perspective of the hidden layers, the activations are changing all the time and run into the problem of covariant shift
    - BN reduces the amount that the distribution shifts about
    - no matter how the inputs activations change, at least the mean and variance will stay relatively the same
    - weakens coupling between jobs of later layers and earlier layers
- Batch Norm as regularization
    - each minibatch is scaled by the mean/variance computed on just that mini-batch
    - this adds some noise to the values z[l] within that minibatch
    - so similar to dropout, it adds some noise to each hidden layers’ activation
    - this has a slight regularization effect
    - the larger the mini-batch size, the smaller the noise and the less regularization

Batch Norm at Test Time

- mui and sigma squared, mean and sd, are computed on the entire mini-batch

- at test time, a minibatch size of examples might not be available
- taking the mean and variance of one example doesn’t make sense
- come up with a separate estimate of mui and sigma squared, using exponentially weighted average (across mini-batches)
- for each mini-batch, run a exponentially weighted average for the mean and sigma^2
- at test time, compute znorm using whatever mui and sigma^2 to do the scaling
- also called a running average

Softmax Regression

- multiple possible classes
- other class as well
- C = number of classes

- Softmax layer
    - activation function
    - t = e^z[l]

    - the output vector is the individual node t divided by the average
    - a = g(z[l]), outputs same shape vector
- Softmax examples
    - NN with no hidden layer
    - decision boundaries will be linear

    - with hidden layers, decision boundaries can be non-linear

Training a Softmax Classifier

- Understanding softmax
    - z is the output layer computation
    - t is a temp variable that performs element wise exponentiation
    - activation function will have output of the softmax

    - hard max is putting 1 in the position of the biggest element as opposed to the probabilities
    - softmax regression generalizes logistic regression to C classes
    - softmax reduces to logistic regression with c = 2
- Loss function
    - the loss is the minus sum of the ylogyhats
    - if the alg want the loss to be small, it wants -logyhat to be small for the correct value, which forces that probability to be the largest in the output vector

    - cost is 1/m sum over loss over all training examples
    - examples will be stacked horizontally […]

- Gradient descent with softmax
    - dZ[l] =yhat - y

Deep learning Frameworks

- large models is not practical to implement from scratch
- more efficient with frameworks
    - Caffe/Caffe2
    - DNTK
    - DL4J
    - Keras
    - Lasagne
    - mxnet
    - PaddlePaddle
    - TensorFlow
    - Theano
    - Torch
- choosing deep learning frameworks
    - ease of programming
    - running speed
    - truly open

TensorFlow

- basic flow of a Tensorflow program
- Motivating problem
    - J = w^2 - 10w + 25 =  (w-25)^2
    - how to minimize?

    ```python
    w = tf.Variable(0, dype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(0.1)

    def train_step():
    	with tf.GradientTape() as tape: # record sequence of operations
    		cost = w ** 2 - 10 * w + 25
    	trainable_variables = [w]
    	grads = tape.gradient(cost, trainable_variables)
    	optimizer.apply_gradients(zip(grads, trainable_variables)) # zip pairs up corresponding elements

    # stepping thought with train_step() gives output of 5
    ```

- in TensorFlow, you only need to implement the forward prop
- the optimization parameter needs to be a tf.Variable(..)
- what is the cost function depends on x and y
- the input numbers x will play the role of the coefficients of the cost function

    ```python
    w = tf.Variable(0, dype=tf.float32)
    x = np.array([1.0, -10.0, 25.0], dtype=np.float32)
    optimizer = tf.keras.optimizers.Adam(0.1)

    def training(x, w, optimizer):
    	def cost_fn():
    		return x[0] * w ** 2 + x[1] * w + x[2]
    	for i in range(1000):
    		optimizer.minimize(cost_fn, [w]) # the [w] syntax is the same as the gradient tape and apply gradients
    	return w

    w = training(x, w, optimizer)

    # converges to 5
    ```

- the heart of a tensorflow program is something to compute the cost
- the return line allow tf to construct a computation graph

- don’t need to explicitly implement backprop

Learn about Gradient Tape and More

- [TensorFlow Course](https://www.coursera.org/specializations/tensorflow-advanced-techniques)