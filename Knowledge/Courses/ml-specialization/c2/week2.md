Train a NN in TensorFlow

- given set of (x,y) examples
- how to build and train this code?
- step 1: specify model that tells TensorFlow how to compute forward inference
- step 2: compile model with specific loss function
- step 3: train the model

Model Training Steps

- logistic regression in first course
    1. specify how to compute output given input x and parameters w,b (define model), f_w,b(x) = ?
        - z = np.dot(w,x) + b
        - f_x = 1/(1+np.exp(-z)
    2. specify loss and cost L(f_w,b(x),y), J(w,b) = 1/m Sum_m^(i=1)L(f_w,b(x^i), y^i)
        - loss = -y * np.log(f_x) - (1-y) * np.log(1-f_x)
    3. train on data to minimize J(w,b)
        - w = w - alpha *dj_dw
        - b = b - alpha *dj_db
- mapping to NN
    1. model = Sequential([…])
    2. model.compile(loss=BinaryCrossentropy()) → binary cross entropy
    3. model.fit(X, y, epochs = 100)

Loss and cost functions

- for handwritten digit classification problem
- most common is the same as for logistic regression

- from statistics the name binary cross entropy is the equation above → logistic loss
- syntax is to ask Tensorflow to compile the NN using this loss function

- having specified the loss with regards to a single training example, Tensorflow knows that the cost you want to minimize is the ticking average over all m training examples of the loss on all the training examples
- optimizing the cost function will result in fitting the NN to the binary classification data
- in case you want to solve a regression problem (more than a couple outputs) rather than a classification problem, tell Tensorflow to compile the model using a different loss function
- for example

- cost function is a function of all the parameters in the NN

Gradient Descent

- repeatedly update W and B

- key thing is the partial derivatives
- what Tensorflow does to compute the gradient → back propagation
- Tensorflow can to all of it for you

    Which line of the code updates the network parameters in order to reduce the cost?

    - [ ]  model.compile(loss=BinaryCrossentropy())
    - [ ]  model = Squential([…])
    - [ ]  none of the above — this code does not update the network parameters
    - [x]  model.fit(X,y,epochs=100)

Demand Prediction Example

- sigmoid activation assumes the features the NN finds are binary, like awareness
- can swap out sigmoid for a different activation function
- a common choice is man(0,z) → ReLU / Rectified Linear Unit

- three most common activation functions are sigmoid, ReLU, and linear activation function which is just g(z) = z or “no activation function”
- there is a forth called softmax activation that will be learned later
- can build a rich variety of NN

Output Layer activation function

- turns out there will be a natural choice based on the target of the ground truth label y
- binary classification → sigmoid
- regression (stock price) → linear / then the output can either be positive or negative
- regression (price of a house) → ReLU / non-negative

Hidden Layer activation function 

- ReLU is by far the most common choice
- sigmoid is really only used int eh output layer when needed
- reason 1: ReLU is faster
- reason 2 (more important): goes flat on one side, sigmoid goes flat on both sides
- GD is slow when the activation function is flat in a lot of places
- GD optimized the cost function J(W,B) but the activation function is a piece of what goes into computing and results in more places in the cost function that flat as well

Choosing Activation Summary

- output layer
    - binary → sigmoid
    - regression (neg/pos) → linear
    - regression (y ≥ 0) → relu

Why do we need activation functions?

- without activation functions, the network would be no different than linear regression
- defeats the entire purpose of using a NN

Linear Example

- if g(z) = z is used everywhere

- equivalent to linear regression
- if output activation is sigmoid → equivalent to logistic regression
- don’t use linear activation in hidden layers

MNIST example

- ten possible digits to recognize → multiclass-classification problem
- target y can take on more than 2 possible values

Softmax

- logistic regression → 2 possible output values

- computes both a_1 and a_2
- softmax regression (4 possible outputs)
- z_1-4 = w_1-4 . x + b_1-4

- for all a_1-4

Softmax regression (N possible outputs)

- z_j = w_j . x + b_j
- parameters w_1, w_2, …, w_n and b_1, b_2, …, b_n

- note: a_1 + a_x + … + a_n = 1
- with n=2, softmax reduces to logistic regression

Cost function for softmax regression

- logistic regression loss can be simplified to

- where the cost function is the average loss
- softmax regression loss

- incentivizes the alg to make a_j as large as possible because whatever the actual value y was, you want the algorithm to say that the chance of y being that value was pretty large
- the loss function, y can take on only one value so you end up computing only on -log(a_j)
- assuming the cost is just the average loss

NN with Softmax output

- 10 classes would require the output layer to have 10 units → softmax layer

- softmax is different computation-wise from other activation functions because you obtain activation values element-wise
- softmax activation for one element requires all zs
- a_1 is a function z1-10 simultaneously

MNIST with softmax

1. specify the model

2. specify loss and cost

    - categorical → still classifying y into categories
    - sparse → output is between ten values, only one of the categories
3. train on data to minimize cost

Numerical Roundoff Errors

- computer only has a finite amount of memory to store each number
- depending on how you decide to compute values, the result could have more or less numerical roundoff error
- more numerically accurate implementation of logistic loss
- if you allow tensorflow to not compute a as an intermediate term, but instead, if you tell tensorflow that the loss is the expression below, just an expansion, then tensorflow can rearrange terms and come up with a more numerically accurate way to compute the loss function

- sets linear activation function for the output layer, and puts the activation function and the crossentropy loss into the specification of the loss function

- logits is z
- downside → becomes less legible

More numerically accurate implementation of softmax

- specify the expansion

- if e^(small number) then the result is an even smaller number, and vice verse for large numbers
- and by rearranging terms, tensorflow can avoid creating these large or small numbers and therefore come up with a more accurate computation for the loss function
- make the last layer linear and make the loss:

MNIST (more numerically accurate)

- with the change, the output layer no longer output the probabilities a_1 → a_10
- instead outputs z_1 → z_10
- so you need to map it through the softmax or logistic function

Multi-label Classification

- associated with a single image x are numerous labels

- the target of the y is a vector of three numbers
- can make three separate NN
- or a single NN to recognize all three

GD

- how to make more efficient?

- is there an alg that will recognize the similar direction of the sequential changes and increase the alpha to take bigger steps to get the minimum faster
- alg called “Adam”
- go faster → increase alpha
- go slower → decrease alpha

Adam Algorithm Intuition

- Adam: Adaptive Moment estimation / not just on alpha
- an alpha for each parameter

- worth trying a few learning rates
- become the standard

Additional Layer Types

- Dense Layer → each neuron’s output is a function of all the activation outputs of the previous layer

- Convolutional Layer
    - neuron only looks at a region of the image
    - faster computation
    - needs less training data (less prone to overfitting)

Convolutional NN

- EKG corresponds to the height of the heart readings as inputs
- have neurons look at only certain windows of inputs
- convolutional layer → each unit only looks at a limited window of the inputs
- next hidden layer could be windows of activations from the previous layers
- output layer would be sigmoid → yes, issue or no, no issue

Derivative Example

- cost function: J(w) = w^2
- say w = 3, J(w) =3^2 = 9
- if we increase w by a tiny amount, epsilon = 0.001, how does J(w) change?
- w = 3 + 0.001
- F(w) = w^2 = 9.006.001
- if w ^ 0.001 → J(w) ^ 6 x 0.001
- roughly 6 x epsilon increase
- derivative w.r.t w of J(w) = 6

Informal Definition of Derivative

- if w ^ epsilon causes J(w) ^ k x epsilon then derivative w.r.t w of J(w) = k

- if the derivative is small, then this update step will make a small update to w_j
    - if the derivative is small, then changing w doesn’t make a difference to the value of j, so let’s not bother to make a change
- if the derivative is large, then this update step will make a large update to w_j
    - if the derivative is large, then even a tiny change to w_j will make a big difference in how much you can change or decrease the cost function J(w), so in that case, lets make a big change because doing so will make a big difference to how much we can reduce the cost function J

More Derivative Examples

- w = 3 → derivative w.r.t w of J(w) = 6
- w = 2 → derivative w.r.t w of J(w) = 4
- w = -3 → derivative w.r.t w of J(w) = -6

- library package sympy can compute the derivatives

A note on derivative notation

- if J(w) is a function of one variable (w),

- if J(w_1,w_2,…,w_n) is an function of more than one variable / “partial derivative”

Small NN Example

- linear activation a = g(z) = z

- computation graph for forward prop

Computing the Derivatives

- back prop
- go back through the computation graph and ask how the output would change if increased by epsilon

Backprop is an efficient way to compute derivatives

- computing derivative of J w.r.t. a once and using it to compute both derivatives of J w.r.t. w and b
- if N nodes and P parameters, compute derivatives in roughly N + P steps rather than N x P steps

Larger NN Example

- if you calculate the derivates (the changes) by the ‘checking method’ (going through forward prop every time per variable), would lead to inefficient NxP rather than efficient backprop N+P
- sometimes done auto-div or auto differentiation

Debugging a learning alg

- implemented regularized linear regression on housing prices
- it makes unacceptably large error in predictions, what to

- get more training examples
- try smaller sets of features
- try getting additional features
- try adding polynomial features (x_1^2, x_2^2, x_zx_2, etc)
- try decreasing lambda
- try increasing lambda
- the key to being effective at how you build a machine learning alg will be if you can find a way to make good choices about where to invest your time

Machine learning diagnostic

- a test that you run to gain insight into what is/isn’t working with a learning alg, to gain guidance into improving its performance
- some of these diagnostics will tell you things thing like, is it worth weeks, or even months collecting more training data, because if it is, then you can then go ahead and make the investment to get more data, which will hopefully lead to improved performance
- diagnostics can take time to implement, but doing so can be a very good use of time

Evaluating a model

- having a systematic way to evaluate performance will paint a clearer path for how to improve the models performance
- model fits the training data well but will fail to generalize to new examples not in the training set

- you can plot and see the overfitting with just two inputs for example, but becomes a lot harder with four or more features
- split the dataset into 70% training set and 30% test set
- test subscript for test examples

Train/test procedure for linear regression (with squared error cost)

- fit parameters by minimizing cost function J(w, b)

- J_train(w,b) will be low
- but J_test(w,b) will be high

- seeing that J test is high on this model, gives you a way to realize that even though it does great on the training set, it is actually not so good at generalizing to new examples to new data points that were not in the training set

Train/test procedure for classification problem

- fit parameters by minimizing J(w,b) to find w,b

- another way is a fraction of the test set and the fraction of the train set that the alg has misclassified

- count y_hat /= y
- J_test(w,b) is the fraction of the test set that has been misclassified