Train / Dev / Test sets

- how to make a NN work well
- Applied ML is a highly iterative process
    - number of layers
    - number of hidden units
    - learning rates
    - activation functions
    - when starting off, almost impossible to correctly guess the right values
    - idea → code → experiment → idea …
    - intuition often does not transfer from one field to another
    - need to try many different models → iterative process
    - how efficiently can you go around the cycle
    - data
        - training set
        - hold-out, cross validation, development set, ‘dev’ set
        - test set
    - often 70%/30% or 60%/20%/20% with test set
    - but now, with big data, dev and test sets are becoming smaller
    - for example, out of one million, 10,000 for both is good, 98 / 1 / 1
- Mismatched train/test distribution
    - training set: cat pictures from webpages
    - dev/test sets: cat pictures from users using an app
    - important to make sure dev and test sets come from same distribution
    - might be okay to not have a test set

Bias/Variance

- high bias → underfit, not a good fit
- complex classifier can fit the data perfectly
- high variance → overfitting

- in high dimensional data, you can plot the graph to understand bias and variance
- look at two errors, train and dev
- overfit, high variance
    - train set error → 1%
    - dev set error → 11%
- underfit, high bias
    - train set error → 15%
    - dev set error → 16%
- high bias and high variance
    - train → 15%
    - dev → 30%
- predicated that the optimal error (Bayes) is nearly zero
- adjust the percentages based on Bayes error (optimal)
- High bias and high variance

Basic Recipe for Machine Learning

- high bias? (training data performance)
    - bigger network
    - train longer
    - NN architecture search
- high variance? (dev set performance)
    - more data
    - regularization
    - NN architecture search
- getting more data if the model has high bias won’t help
- bias variance tradeoff → pre deep learning
- bigger network and more data are the tools to improve high bias or high variance on their own

Regularization

- Logistic regression
    - minimize cost function J
    - add regularization with additional term lambda/2m * ||w||^2_2
    - L2 norm is w.Tw

    - b regularization is omitted because w is high dimensional whereas b is a single number
    - not much of a difference
    - L2 is most common
    - L1 regularization

    - w will be sparse → a lot of zeros, help with compression model, not used much
    - lambda is the regularization hyperparameter
    - set using dev set
    - lambda is a reserved keywork in python, use lambd
- NN
    - same but with squared norm
    - known and Frobenius norm, sum of squared elements of a matrix

    - how to use GD

    - L2 regularization also known as weight decay

    - also multiplying w by something less than one

Why Regularization Reduces Overfitting?

- regularization penalizes weight matrices from being too large
- the weights are incentivized to be close to zero because the cost is now greater if lambda is huge
- you are effectively zeroing out a lot of nodes and the model becomes simpler
- if z is small in the tanh function, only the linear regime is being using
- when z is gets larger or small, the activation function becomes less linear
- large lambda → small weights
- is weights are small, then z[l] = w[l] * a[l-1] + b[l], every layer will be roughly linear
- deep network with linear layers cannot fit nonlinear decision boundaries

- one way of debugging a model is plot the cost function J over iterations
- so plot with regularization term

Dropout Regularization

- if a network is overfitting, set some probability of eliminating a node in NN

- different nodes from each training example
- Implementing dropout (”inverted dropout”)
    - illustrate with layer l=3
    - d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep.prob
    - a3 = np.multiply(a3, d3)
    - a3 /= keep.prob → corrects expected value of z
- Making predication at test time
    - no drop out at test time → adds noise
    - could runs predictions many times with dropout and then average but computationally inefficient and similar result

Understanding Dropout

- Why does drop-out work?
    - intuition: every iteration, the work is done on a smaller NN, so almost like a regularizing effect
    - cant rely on any one feature, so have to spread out weights
    - shrinks the squared norm of the weights
    - vary keep-prob by layer, the higher node count the higher probability, 0.5
    - can do on the inputs, but not used very much
    - useful in computer vision
    - J will no longer be less well defined
    - so turn off dropout, check that J is decreasing, then turn dropout back on

Other Regularization Methods

- Data augmentation
    - mirrors, flipping
    - random crops
    - distortions and translations
    - flip vertically is not good for cats because you don’t want upside-down cats
- Early stopping
    - plot the training error
    - also plot the dev set error
    - stops training when dev set error starts to diverge
    - before training for too long, w isn’t so big
    - by picking a NN with smaller norm, the NN will overfit less
    - not as good because the tools for optimizing the cost function J should all you care about
    - a separate task is reducing overfitting
    - early stopping couples the two tasks and breaking how J is being optimized
    - just use L2 regularization
    - search space of hyperparameters is easier to decompose
    - downside is trying many values of the regularization parameter lambda
    - where early stopping gets to try out values of small w, mid w, and large w, without needing to try a lot of values of the L2 regularization hyperparameter

Normalizing Inputs

- Normalizing training sets
    - zero out mean

    - normalize variances
    - x1 has more variance

    - use the same mui and sigma to normalize the test set
- Why normalize inputs?
    - cost function will look squished out

    - works if they are dramatically different scales

Vanishing / Exploding Gradients

- problem is how the derivates of slope can be very big or very small which makes training difficult
- activation values will decrease exponentially as a function of the depth of the network
- explained using linear activation on a very deep network
- the more layers, the more the activations stack and grow

- similar argument for derivatives
- tiny gradient will take GD a long time

Weight Initialization for Deep Networks

- partial solution is a more careful initialization of the weights on the network
- Single neuron example
    - larger n → smaller w_i should be
    - set var(w_i) = 1/n
    - W[l] = np.random.randn(shape) X np.sqrt(2/n^[l-1])
    - with relu activation, 2/n work better ^
    - trying to set each of the weight matrices w so that it’s not too much bigger than 1 and not too much less than 1
    - doesn’t solve, but help reduce the issue
    - tanh → sqrt(1/n) is better, Xavier initialization
    - tanh → sqrt(2/(n[l-1] _ n[l]))
    - 2 or 1 could another hyperparameter

Numerical Approximation of Gradients

- Checking the derivative computation’
    - given f(theta) = x^3
    - nudge theta to the left and right
    - height over width of a bigger distance on a curve gives a better approximation

    - taking the slope is approximately g(theta), the derivative

    - approx error: 0.0001
    - opposed to the one sided difference of just plus epsilon with error of 3.0301
    - both ways to approx
    - greater confidence that g(theta) is the correct implementation of the derivative
    - the error is on the order of O(epsilon^2)
    - whereas the other formula is on the order of O(epsilon) which is greater

Gradient Checking

- how to verify if the implementation of back prop is correct
- Gradient check for a NN
    - take all parameters and reshape into a big vector theta, concatenate
    - take all the derivatives and reshape into a big vector dtheta
    - is dtheta the gradient of J(theta)
- Gradient checking (Grad check)
    - loop for each component of theta
        - dtheta_approx[i] = J(theta1, theta2, …, theta_i + epsilon, …) - J(theta1, theta2, …, theta_i - epsilon, …) / 2theta
        - should be approx equal to dtheta[i]
    - check if dtheta_approx is reasonably close to dtheta
    - check the Euclidian distance between the two difference, L2 norm, and divide by dtheta_approx + dtheta like regularization
    - if you get 10^-7 → great
    - if you get 10^-5 → maybe okay
    - if you get 10^-3 → bug
    - look at individual components of dtheta_approx

Gradient Checking Implementation Notes

- don’t use in training - only to debug
- slow computation
- look at components to try to identify bug, which values of i, components of theta
- all values hone in on db or dw then you check respective part of the implementation
- remember regularization, add when using regularization
- doesn’t work with dropout, run without drop out, and hope the implementation of dropout is correct
- run at random initialization; perhaps again after some training

Yoshua Bengio Interview

- long term dependencies
- auto-encoders
- piecewise linear activation function
- GAN
- attention makes translation work
- Jeff Hinton
- supervised learning → requires humans to define what the important concepts are for the problem and to label those concepts in the data
- unsupervised learning → building a mental construction that explains how the world works by observation
- has a book: https://www.deeplearningbook.org/front_matter.pdf
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025