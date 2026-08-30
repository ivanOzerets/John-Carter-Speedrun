Binary Classification

- when you’re implementing a NN you usually don’t want an explicit loop over all of the training set
- forward prop and then backward pass
- logistic regression is an alg for binary classification
- unroll all pixel intensity values into one array → feature vector
- if 64x64, then 64x64x3=12288
- n = n_x = 12288

- Notation
    - single training example → (x,y) | x is in R^(n_x), y is {0,1}
    - m training examples: {(x^1, y^1), …, (x^m, y^m)}
    - m=m_train or m_test=#of test examples
    - compact notion for training examples → into columns

    - convenient to stack output variables as well

Logistic Regression

- given input feature vector x, want y_hat = P(y=1 | x)
- you want y_hat to tell you the chance the input image is a cat image
- parameters: w is in R^(n_x), b is in R
- output y_hat = w^Tx + b, w is transposed so the columns match the rows
- but this isn’t a good algorithm because you want true or false, between 0 and 1
- difficult to enforce because w^Tx + b can be much bigger than one or negative
- cast it onto the sigmoid function y_hat = sigma(w^Tx + b)

- z = w^Tx + b
- sigmoid(z) = 1/(1+e^-1)
- if z is large then sigmoid(z) ~ 1
- if z is large negative then sigmoid(z) ~ 0
- job is to learn parameters w and b so that y_hat becomes a good estimate of the chance of y being equal to one
- w and b are kept separate usually
- alternative notation has x_0 = 1 and y_hat = sigmoid(theta^Tx)

Logistic Regression Cost Function

- to train the model parameters you need a cost function
- you want the predicted outputs to be close to the true outputs
- superscript parenthesis are associated with data, ith example
- loss (error) function are used to measure how well the alg is doing
- half the square distance is not optimal because sigmoid becomes non convex when optimizing
- loss if a function of how good y_hat is from y
- the loss function used is -(ylog(y_hat) + (1-y)log(1-y_hat)) (-log() below)

- loss function is how you are doing on one example
- the cost function is how you are doing on the whole training set

Gradient Descent

- cost function measures how well parameters w and b are doing on the training set
- b is usually initialized to zero

- in the direction of steepest descent
- repeat the loop
    - set w equal to w - alpha x gradient of w
    - set b equal to b - alpha x gradient of b
- alpha is the learning rate
- dw will be used as the variable name for the gradient of the cost function wrt w
- more than one variables in function → partial derivative
- one variable → regular derivative

Derivatives

- Intuition about derivatives
    - slope of a function is the derivative
    - if you nudge one axis a little bit, how much does the other change wrt the first

- df(a)/da = 3

More Derivative Examples

- Intuition about derivatives
    - now if the function is exponential
    - nudging a little on one axis will double value on the other axis
    - ratio of the slope changes

    - derivative of log(a) = 1/a

Computation Graph

- J(a,b,c) = 3(a + bc)
- u = bc
- v = a + u
- J = 3v

- comes in handy when there is a special output variable, such as J, that is being optimized
- left to right pass ^
- computing derivative becomes the right to left pass

Derivatives with a Computation Graph

- Computing derivatives (on above graph)
    - how does the end product get changes wrt changes in computation previous
    - dJ/dv = 3
    - dJ/da = 3, because dv/da = 1
    - if changing a affects change in J through v, chain rule
    - dJ/db
    - notation of variables for derivatives will be the derivative of the final output variable of J, wrt the various intermediate quantities
    - dJ/du = 3
    - dJ/db = dJ/du x du/db = 6
    - dJ/db = 9

Logistic Regression Gradient Descent

- Logistic regression recap
    - z = wTx + b
    - y_hat = a = sigma(z)
    - loss(a,y) = -(ylog(a) + (1-y)log(1-a))

- Logistic regression derivatives
    - da = -y/a + (1-y)(1-a)
    - dz = a - y
    - dw1 = x1dz
    - dw2 = x2dz
    - db = dz

Gradient Descent on m Examples

- Logistic regression on m examples
    - a^i = y_hat^i = sigma(z^i) = sigma(wTx^i + b)
    - with one training example ^
    - the derivative of the cost function will be the average of the individual loss terms
    - so compute the derivatives of all the losses and then average them
    - n is the number of features
    - m is the number of training examples

    - dw1 = 0, dw2 = 0, db = 0 will be used as accumulators, no superscript
    - above alg has two loops, training examples and then features
    - less efficient than without explicit for loops → vectorization

Vectorization

- getting rid of explicit for loops

- both cpu and gpu have parallelization instructions → SIMD - single instruction multiple data

More Vectorization Examples

- NN programming guideline
    - whenever possible, avoid explicit for-loops

- Vectors and matrix valued functions
    - say you need to apply the exponential operation on every element of a matrix/vector

- Logistic regression derivatives
    - initialize zeros with np.zeros()
    - looping over features can be done with straight multiplication
    - averages can be done with one division

Vectorizing Logistic Regression

- no explicit for loops
- Vectorizing Logistic Regression
    - X is (nx, m) matrix
    - first construct a 1xm matrix to compute all z’s at the same time
    - wTX + [b,b,…b] (1xm) = [wTx(i) + b, .., ]
    - training examples stack are represented with capital letters
    - np.dot(w.T, x) + b
    - A = [a^1, … a^m] = sigma(z)

Vectorizing Logistic Regression’s Gradient Output

- Vectorizing Logistic Regression
    - A - Y for dz = a - y
    - before we initialized dw to 0 and then looped over the training examples by adding x^idz^i
    - dz because of chain rule
    - same for db and then average by dividing by m

- Implementing Logistic Regression
    - the loops are the training examples and the parameters
    - the eliminate the loops, parameters can be multiplied and added with the dot product np.dot
    - you perform the loops sequentially, just all at once
    - Z = wTX + b → np.dot(wT,X) + b
    - A = sigma(z)
    - dZ = A - Y
    - dw = (1/m)XdZT
    - db = (1/m)np.sum(dz)
    - w := w-alphadw (:= is assignment)
    - b := b-alphadb

Broadcasting in Python

- Broadcasting example
    - sum columns and give percentages of each value from the total → no loops

    - cal = A.sum(axis=0) (axis = 0 is to sum vertically)
    - perc = 100*A/cal.reshape(1,4) (reshape is overkill since cal is already a 1x4)
    - reshape is very cheap in constant time so don’t be shy and use it
    - 3/4 matrix is divided by a 1/4 matrix
    - python auto expands a constant, column vectors and row vectors

- General Principle
    - an operation applied to a m,n matrix with a 1,n or m,1 matrix will turn the 1 into the respective m or n

A Note on Python/Numpy Vectors

- python and numpy can have strange bugs
- a = np.random.randn(5) → creates five random gaussian variables
- print(a.shape) → ‘rank one’ array (5,), neither a column nor a row vector
- transpose and dot with transpose of itself give the same values back and a number respectively
- recommended to not use rank one array
- instead → a = np.random.randn(5,1)
- Python/numpy vectors
    - rank one arrays do not behave like a column or row vectors
    - you can use an assert statement
    - assert(a.shape == (5,1))
    - you can also reshape with a = a.reshape((5,1))

Quick tour of Jupyter/iPython Notebooks

- shift+enter to run code block
- code runs on a kernal, it might die after a while → restart
- run sequential blocks of code

Explanation of Logistic Regression Cost Function

- Logistic regression cost function
    - we want the alg to output y_hat as the chance that y=1 for a given set of input features x
    - if y=1 : p(y | x) = y_hat
    - if y=0 : p(y | x) = 1 - y_hat
    - can combine with p(y | x) = y_hat^y x (1 - y_hat)^(1 - y)
    - because log is a strictly monotonically increasing function, maximizing log(p(y | x)) gives similar results as optimizing P(y | x)

    - minimizing the loss corresponds to maximizing the log of the probability
- Cost on m examples
    - p(labels in training set) = product (p(y^i | x^i)
    - principle of maximum likelihood estimation says to choose parameters that maximize the negative loss

    - works if the training examples are IID → identically independently distributed

Pieter Abbeel Interview

- deep learning relaxes the domain knowledge requirement
- supervised learning is for learning input output mapping
- reinforcement learning is the notion of where does the data come from, how do you do credit assignment, and issues of safety
- deep part is the representation
- if there is a pattern, it can be probably represented with a deep network to capture that pattern
- deep reinforcement learning has very short horizon
- how to get systems to reasons over long time horizons
- Andrew Karpathy
- Berkley deep learning course