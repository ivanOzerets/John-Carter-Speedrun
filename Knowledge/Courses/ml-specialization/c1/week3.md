Motivation
- classification → output variable y can take on only one of a small handful of values instead of any number
- one of two possible categories of classes → binary classification
- output can be no/yes, false/true, 0/1, negative class/positive class
- not good or bad → absence or presence
- logistic regression is used for classification → confusing for historical reasons

### Optional Lab: Classification

Classification Problems
- examples of classification problems are things like: identifying email as spam or not spam or determining if a tumor is malignant or benign
- in particular, these are examples of binary classification where there are two possible outcomes
- outcomes can be described in pairs of positive/negative
- plots of classification data sets often use symbols to indicate the outcome of an example
- the plots below, ‘X’ is used to represent the positive values while ‘O’ represents negative outcomes
    ```python
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])
    ```
    ```python
pos = y_train == 1
neg = y_train == 0

fig,ax = plt.subplots(1,2,figsize=(8,3))
#plot 1, single variable
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlc["dlblue"],lw=3)

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()

#plot 2, two variables
plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()
plt.show()

    ```
- in the single variable plot, positive results are shown both a red ‘X’s and as y=1
- negative results are blue ‘O’s and are located at y=0
    - recall in the case of linear regression, y would not have been limited to two values but could have been any value
- in the two-variable plot, the y axis is not available
- positive results are shown as red ‘X’s, while negative results use the blue ‘O’ symbol
    - recall in the case of linear regression with multiple variables, y would not have been limited to two values and a similar plot would have been three-dimensional

Linear Regression approach
- lets try to apply linear regression to build a prediction model approach using a simple example
- the model will predict if a tumor is benign or malignant based on tumor size
    - find the best linear regression model for the given data
        - the resulting linear model does not match the data well
        - one option to improve the results is to apply a threshold
    - toggling 0.5 threshold will show the predictions if a threshold is applied
        - these predictions look good, the predictions match the data
    - adding further ‘malignant’ data points on the far right, in the large tumor size range (near 10)
        - model predicts the larger tumor, but data point at x=3 is being incorrectly predicted 
    ```python
w_in = np.zeros((1))
b_in = 0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)
    ```
- the example above demonstrates that the linear model is insufficient to model categorical data

Logistic regression
- most widely used classification alg in the world
- fit an s-shaped curve
- sigmoid function / logistic function / outputs between 0 and 1
- logistic regression model
- passing value of straight value function as z to sigmoid function g(z)

Interpretation of logistic regression output
- think of the output of logistic regression as returning the probability that class is 1
- example
    - x is ‘tumor size’
    - y is 0 (not malignant) or 1 (malignant)
    - f_w,b(x) = 0.7 → 70% chance that y is 1
- if the output can only be 0 or 1 and y has a 70% chance of being 1, what is the chance that y is 0?
    - there is a 30% chance → P(y=0) + P(y=1( = 1
- f_w,b(x) = P(y=1 \| x; w,b) → probability that y is 1, given input x, parameters w, b

### Optional Lab: Logistic Regression

Sigmoid or Logistic Function
- for a classification task, we can start by using our linear regression model, f_w,b(x^i) = w.x^i + b, to predict y given x
    - however, we would like the predictions of our classification model to be between 0 and 1 since out output variable y is either 0 or 1
    - this can be accomplished by using a ‘sigmoid function’ which maps all input values to values between 0 and 1

Formula for Sigmoid function
- the formula for a sigmoid function is as follows - g(z) = 1/(1+e^-z)
- in the case of logistic regression, z (the input to the sigmoid function) , is the output of a linear regression model
    - in the case of a single example, z is scalar
    - in the case of multiple examples, z may be a vector consisting of m values, one for each example
    - the implementation of the sigmoid function should cover both of these potential input formats
- numpy has a function called exp(), which offers a convenient way to calculate the exponential (e^z) of all elements in the input array (z)
- it also works with a single number as an input 
    ```python
# Input is an array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

# Input to exp: [1 2 3]
# Output of exp: [ 2.72  7.39 20.09]
# Input to exp: 1
# Output of exp: 2.718281828459045
    ```
- the sigmoid function implemented in python 
    ```python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g
    ```
    ```python
# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Input (z), Output (sigmoid(z))
# [[-1.000e+01  4.540e-05]
#  [-9.000e+00  1.234e-04]
#  [-8.000e+00  3.354e-04]
#  [-7.000e+00  9.111e-04]
#  [-6.000e+00  2.473e-03]
#  [-5.000e+00  6.693e-03]
#  [-4.000e+00  1.799e-02]
#  [-3.000e+00  4.743e-02]
#  [-2.000e+00  1.192e-01]
#  [-1.000e+00  2.689e-01]
#  [ 0.000e+00  5.000e-01]
#  [ 1.000e+00  7.311e-01]
#  [ 2.000e+00  8.808e-01]
#  [ 3.000e+00  9.526e-01]
#  [ 4.000e+00  9.820e-01]
#  [ 5.000e+00  9.933e-01]
#  [ 6.000e+00  9.975e-01]
#  [ 7.000e+00  9.991e-01]
#  [ 8.000e+00  9.997e-01]
#  [ 9.000e+00  9.999e-01]
#  [ 1.000e+01  1.000e+00]]
    ```
- the values in the left column are z, and the values in the right column are sigmoid(z)
- the input values to the sigmoid range from -10 to 10, and the output values range from 0 to 1
- plot the function using matplotlib library 
    ```python
# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
    ```
- the sigmoid function approaches - as z goes to large negative values and approaches 1 as z goes to large positive values

Logistic Regression
- a logistic regression model applies the sigmoid to the familiar linear regression model
- apply logistic regression to the categorical data example of tumor classification
- load the examples and initial values for the parameters 
    ```python
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0
    ```
- orange line is ‘z’ or w . x^i + b
- it does not match the line in a linear regression model
- you can further improve these results by applying a threshold
- if you add further data points in the large tumor size range, and re-run logistic regression the model continues to make correct predictions

Decision Boundary
- the sigmoid function is above the 0.5 boundary when z is above 0
- if there is more than one input variable, then the sigmoid function can still be used
- decision boundary → z = w . x + b = 0 → z = x_1 + x_2 - 3 = 0 → x_1 + x_2 = 3 (for example)

Non-linear decision boundaries
- f_w,b(x) = g(z) = g(w_1x_1^2 + w_2x__2^2 + b)
- decision boundary → z = x_1^2 +x_2^2 - 1 = 0 → x_1^2 + x_2^2 = 1
- can make even more complex decision boundaries with more input terms
- ellipse or something funky

### Optional Lab: Logistic Regression, Decision Boundary

Dataset
- suppose you have the following training dataset
    - the input variable X is a numpy array which has 6 training examples, each with two features
    - the output variable y is also a numpy array with 6 examples, and y is either 0 or 1

Plot data
- helper function will be used to plot this data
- the data points with label y=1 are shown as red crosses, while the data points with label y=0 are shown as blue circles
    ```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()
    ```

Logistic regression model
- suppose you’d like to train a logistic regression model on this data which has the form f(x) = g(w_0x_0 + w_1x_1 + b) where g(z) = 1/(1+e^-z), which is the sigmoid function
- let’s say that you trained the model and get the parameters as b = -3, w_0 = =1, w_1 = 1 → f(x) = g(x_0 + x_1 - 3) (the parameters will be fit the data later, assuming GD is different)
- understand what this trained model is predicting by plotting its decision boundary

Refresher on logistic regression and decision boundary
- recall that for logistic regression, the model is represented as f_w,b(x^i) = g(w . x^i + b)
- where g(z) is known as the sigmoid function and it maps all input values to values between 0 and 1: g(z) = 1/(1+e^-z)
- and w . x is the vector dot product: w . x = w_0x_0 + w_1x_1
- We interpret the output of the model f_w,b(x) as the probability that y = 1 given x and parameterized by w and b
    - therefore, to get a final prediction (y=0 or y=1) from the logistic regression model, we can use the following heuristic
        - if f_w,b(x) ≥ 0.5, predict y=1
        - if f_w,b(x) \< 0.5, predict y=0
- plot the sigmoid function to see where g(z) ≥ 0.5
    ```python
# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
    ```
- as you can see, g(z) ≥ 0.5 for z ≥ 0
- for a logistic regression model, z = w . x + b
    - if w . x + b ≥ 0, the model predicts y = 1
    - if w . x + b \< 0, the model predicts y = 0

Plotting decision boundary
- out logistic regression model has the form f(x) = g(-3 + x_0 + x_1)
- this model predicts y = 1 if -3 + x_0 + x_1 ≥ 0
- plot the equation, which is equivalent to x_1 = 3 - x_0
    ```python
# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
    ```
- the blue line represents the line x_0 + x_1 - 3 = 0 and it should intersect the x_1 axis at 3 (if we set x_1 = 3, x_0 = 0) and the x_0 axis at 3 (if we set x_1 = 0, x-0 = 3)
- the shaded region represent -3 + x_0 + x_1 \< 0
- the region above the line is -3 + x_0 + x_1 \> 0
- any point in the shaded region (under the line) is classified as y=0
- any point on or above the line is classified as y=1
- the line is known as the decision boundary
- by using higher order polynomial terms, we can come up with more complex non-linear boundaries

Cost function for logistic regression
- squared error cost is not ideal
Logistic loss function
- plot when y = 1 for intuition
- loss function measures how well you do on one training example
- summing all the losses then gets you the cost function
- if the output predicted is close to 1 and the true label is 1 then the loss is pretty much 0
- loss is lowest when f_w,b(x^i) predicts close to true label y^i
- the second part of the loss function
- the further prediction f_w,b(x^i) is from target y^i, the higher the loss
- when the true label is 1, the alg is strongly incentivized not to predict something close to zero 

Cost
- the cost will then be
- find w, b that minimize J

### Optional Lab: Logistic Loss

Squared error for logistic regression
- linear regression we have used the squared error cost function
- the squared error cost had the nice property that following the derivative of the cost lead to the minimum
- this cost function worked well for linear regression, it is natural to consider it for logistic regression
- f_w,b(x) now has a non-linear component, the sigmoid function: f_w,b(x^i) = sigmoid(wx^i + b)
- let’s try squared error cost on the example, including the sigmoid
    ```python
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)
    ```
- surface plot of the cost using a squared error cost:
    ```python
plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()
    ```
- the surface above not nearly as smooth as the soup bowl from linear regression
- logistic regression requires a cost function more suitable to its non-linear nature

Logistic Loss Function
- uses a loss function more suited to the task of categorization where the target is 0 of 1 rather than any number
    - loss is a measure of the difference of a single example to its target value
    - cost is a measure of the losses over the training set
- loss(f_w,b(x^i),y^i) is the cost for a single data point
- f_w,b(x^i) is the model’s prediction, while y^i is the traget value
- f_w,b(x^i) = g(w . x^i + b) where function g is the sigmoid function
- notational convention: log means the natural log
- the defining feature of the loss function is the fact that it uses two separate curves
- one for the case when the target is zero and another for when the target is one
- combined, these curves provide the behavior useful for a loss function,  namely, being zero when the prediction matches the target and rapidly increasing in value as the prediction differs from the target
- combined, the curves are similar to the quadratic curve of the squared error loss
- the x-axis is f_w,b which is the output of a sigmoid
- the sigmoid output is strictly between 0 and 1
- the loss function can be rewritten to be easier to implement
- consider y^i can have only two values, 0 and 1, then you can consider the equation in two pieces
- when y^i = 0, the left-hand term eliminates
- when y^i = 1, the right-hand term eliminates
- with this new logistic loss function, a cost function can be produced that incorporates the loss from all the examples
- the cost vs parameters curve for the simple example
- this curve is well suited to GD
- it does not have plateaus, local minima, or discontinuities
- it is not a bowl as in the case of squared error
- the cost and the log of the cost are plotted to illuminate the fact that the curve, when the cost is small, has a slope and continues to decline

Simplified Loss Function

Simplified Cost Function
- this cost function is derived from stats by maximum likelihood
- convex function

### Optional Lab: Cost function for logistic regression

Dataset
- same dataset as was used in the decision boundary lab
- a helper function to plot this data will be used
- data points with label y=1 are shown as red crosses, while the data points with label y=0 are shown as blue circles
    ```python
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)
    ```
    ```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
    ```

Cost function
- loss is defined to apply to one example
- combine the losses to form the cost, which includes all the examples
- for logistic regression, the cost function is of the form
- where loss(f_w,b(x^i),y(^i) is the cost for a single data point
- where m is the number of training examples in the data set and

Code Description
- the alg for compute_cost_logistic loops over all the examples calculating the loss for each example and accumulating the total
- note that the variables X and y are not scalar values but matrices of shape (m,n) and (m,) respectively, where n is the number of features and m is the number of training examples
```python
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost
```
```python
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

# 0.36686678640551745
```

Example
- cost function output for a different value of w
    - previously, the decision boundary for b = -3, w_0 = 1, w_1 = 1 was plotted
    - let’s say you want to see if b = -4, w_0 = 1, w_1 = 1 provides a better model
- plot the decision boundary for these two different b values to see which one fits the data better
    - for b = -3, w_0 = 1, w_1 = 1, we’ll plot -3 + x_0 + x_1 = 0 (blue)
    - for b = -4, w_0 = 1, w_1 = 1, we’ll plot -4 + x_0 + x_1 = 0 (magenta)
    ```python
import matplotlib.pyplot as plt

# Choose values between 0 and 6
x0 = np.arange(0,6)

# Plot the two decision boundaries
x1 = 3 - x0
x1_other = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0,x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()
    ```
- magenta is a worse model for the training data
- does the cost function implementation reflect this
```python
w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))

# Cost for b = -3 :  0.36686678640551745
# Cost for b = -4 :  0.5036808636748461
```
- the cost function behaves as expected and the cost for magenta is indeed higher than the cost for blue

Training logistic regression
- given new x, output of the model f_w,b(x) can make a prediction and estimate the probability that the label y = 1

GD
- cost function
- may look the same as linear regression
- is not because the definition for the function of f_w,b(x) has changed
- same concepts
    - monitor gradient descent (learning curve)
    - vectorized implementation
    - feature scaling

### Optional Lab: Gradient Descent for Logistic Regression
Dataset
- same dataset as was used in the decision boundary lab
- a helper function to plot this data will be used
- data points with label y=1 are shown as red crosses, while the data points with label y=0 are shown as blue circles
    ```python
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)
    ```
    ```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
    ```

Logistic Gradient Descent
- the gradient descent alg utilized the gradient calculation
- where each iteration performs simultaneous updates on w_j for all j, where
    - m is the number of training examples in the data set
    - f_w,b(x^i) is the model’s prediction, wile y^i is the target
    - for a logistic regression model
        - z = w . x + b
        - f_w,b(x)= = g(z) where g(z) is the sigmoid function g(z) = 1/(1+e^-z)

GD Implementation
- the GD alg implementation has two components
    - the loop implementing the first convergence equation
        - this is gradient_descent below and is generally provided to you in optional and practice labs
    - the calculation of the current gradient
        - this is compute_gradient_logistic below

Calculating the Gradient, Code Description
- Implements the gradient equations for all w_j and b
- many ways to implement this
    - initialize variables to accumulate dj_dw and dj_db
    - for each example
        - calculate the error for that example g(w . x^i + b) - y
        - for each input value x_j^i in this example
            - multiply the error by the input x_j^i and add to the corresponding element of dj_dw
        - add the error to dj_db
    - divide dj_db and dj_dw by total number of examples (m)
    - note that **x**^i is numpy X[i,:] or X[i] and x_j^i is X[i,j]
    ```python
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw  
    ```
    ```python
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )

# dj_db: 0.49861806546328574
# dj_dw: [0.498333393278696, 0.49883942983996693]
    ```

GD Code
- locate and compare the functions in the routine to the equations above
    ```python
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing

    ```
    ```python
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

# Iteration    0: Cost 0.684610468560574   
# Iteration 1000: Cost 0.1590977666870456   
# Iteration 2000: Cost 0.08460064176930081   
# Iteration 3000: Cost 0.05705327279402531   
# Iteration 4000: Cost 0.042907594216820076   
# Iteration 5000: Cost 0.034338477298845684   
# Iteration 6000: Cost 0.028603798022120097   
# Iteration 7000: Cost 0.024501569608793   
# Iteration 8000: Cost 0.02142370332569295   
# Iteration 9000: Cost 0.019030137124109114   

# updated parameters: w:[5.28 5.08], b:-14.222409982019837
    ```
- plotting the results of gradient descent
    ```python
fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability 
plt_prob(ax, w_out, b_out)

# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()
    ```
- the shading reflects the probability y=1 (result prior to decision boundary)
- the decision boundary is the line at which the probability = 0.5

Another Data Set
- using a one variable data set
- with just two parameters, w , b, it is possible to plot the cost function using a contour plot to get a better idea of what GD is up to
    ```python
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
    ```
- we’ll use a helper function to plot this data
- the data points with label y=1 are shown as red crosses, while the data points with label y=0 are shown as blue circles
    ```python
fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()
    ```
    ```python
w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )
    ```

### Optional Lab: Logistic Regression using Scikit-Learn

Dataset
- same dataset as before
    ```python
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
    ```

Fit the model
- the code imports the logistic regression model from scikit-learn
- can fit this model on the training data by calling fit function
    ```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)

# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='auto', n_jobs=None, penalty='l2',
#                    random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
#                    warm_start=False)
    ```

Make Predictions
- you can see the predictions made by this model by calling the predict function
    ```python
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)

# Prediction on training set: [0 0 0 1 1 1]
    ```

Calculate accuracy
- calculate the accuracy of this model by calling the score function
```python
print("Accuracy on training set:", lr_model.score(X, y))

# Accuracy on training set: 1.0
```

The Problem of Overfitting
- does not fit the training set well → high bias / underfit
- a clear pattern in the training data that the algorithm is just unable to capture
- preconception (bias) that the housing prices will be a linear function to the size
- this bias that the data is linear causes the alg to fit a straight line to fit the data and underfit the data
- fits training set pretty well → generalization / just right
- when you overfit you overcomplicate the alg
- on one hand it does an extremely good job because it passes through all the training data perfectly, even actually getting the cost to zero
- this type of model fits the data to well → overfit / high variance

Classification
- overfitting also applies
- underfitting / high bias
- generalized / just right
- overfit / high variance

Addressing Overfitting
- one way to address overfitting is to collect more training data
- select features to include/exclude
- all features + insufficient data → overfit
- selected features → just right / feature selection / use intuition
    - disadvantage is that the alg is throwing away information needing for predicting pricing well

Regularization
- reduce the size of parameter w_j
- more gently reduce the influence of a feature than something as harsh as eliminating it outright
- encourage the learning alg to shrink the weights of the parameters without demanding them to be set to zero
- lets you keep all the features but prevents them from having an overly large effect
- by convention, the size is reduced of w_j parameters, not b, you can, but usually not

### Optional Lab: Overfitting

Overfitting
- categorical
                        - regression
                        - the ‘ideal’ curves represent the generator model to which noise was added to achieve the data set
- ‘fit’ does not use pure gradient descent to improve speed, these methods can be used on smaller data sets

Cost Function with Regularization
- nearly canceling out the influence of w_3 and w_4

Regularization
- small values of w and b → simpler model, less likely to overfit
- if you have a lot of features, say 100, you will not know which ones are important and which ones to penalize
- so to implement regularization, you would penalize all the w_j parameters
- regularization parameter → lambda
- by convention, lambda is divided by 2m
- by scaling both terms by 2m it becomes easier to choose a good value for lambda
- even if the training size grows, the same value of lambda will work, if you have this extra scaling by 2m
- will not penalize b → very little difference
- if lambda is zero → model will overfit
- if lambda is very large → model will underfit
- tradeoff between minimizing the mean squared error and keeping the parameters small

Regularized linear regression
- GD was previously calculated without the regularization term
- the only difference is in the gradient of the cost function w.r.t w_j
- we don’t have to regularize b
- to implement GD
- the update is the same as before except the w_j is multiplied by an extra term
- shrinks the parameters of w_j a little bit at every iteration
- calculation of the derivative term  

Regularized logistic regression
- regularized the same way as linear regression → adding the extra term to the cost function
- the gradients are the same form with the added term on the partial of the cost function w.r.t. w_j

### Optional Lab: Regularized Cost and Gradient

Adding regularization
- cost → the cost functions differ significantly between linear and logistic regression, but adding regularization to the equations is the same
- gradient → the gradient functions for linear and logistic regression are very similar, only differ in the implementation of f_w,b

Cost functions with regularization
- cost functions for regularized linear regression
    - the equation for the cost function regularized linear regression is
- the difference to the cost function without regularization is the regularization term at the end
- including this term encourages gradient descent to minimize the size of the parameters
- the parameter b is not regularized → standard practice
- implementation of equations above
    ```python
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot
        cost = cost + (f_wb_i - y[i])**2                               #scalar             
    cost = cost / (2 * m)                                              #scalar  
 
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar
    ```
    ```python
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

# Regularized cost: 0.07917239320214275
    ```

Cost function for regularized logistic regression
- for regularized logistic regression, the cost function is of the form
- once again, the only difference is the regularization term
- including this term encourages GD to minimize the size of the parameters
- parameter b is not regularized → standard practice
    ```python
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar
    ```
    ```python
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

# Regularized cost: 0.6850849138741673
    ```

GD with regularization
- the basic alg for running GD does not change with reg
- what changes with reg is computing th gradients

Computing the gradient with reg (both linear/logistic)
- the gradient calc for both linear and logistic regression are nearly identical, differing only in computation of f_w,b
    - m is the number of traning examples in the data set
    - f_w,b(x^i) is the model’s prediction, while y^i is the target
    - for a linear regression model → f_w,b(x) = w . x +b
    - for a logistic regression model → z = w . x + b, f_w, b(x) = g(z), where g(z) is the sigmoid function g(z) = 1/(1+e^-z)
    - the term which adds reg is the alpha * (w_j / m)

Gradient function for reg linear regression
```python
def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]                 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]               
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m   
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw
```
```python
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

# dj_db: 0.6648774569425726
# Regularized dj_dw:
#  [0.29653214748822276, 0.4911679625918033, 0.21645877535865857]
```

Gradient function for regularized logistic regression
```python
def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

```
```python
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

# dj_db: 0.341798994972791
# Regularized dj_dw:
#  [0.17380012933994293, 0.32007507881566943, 0.10776313396851499]
```
                
### Lab: Logistic Regression

Logistic Regression
- build a logistic model to predict whether a student gets admitted into a university
- problem statement
    - suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams
        - you have historical data from previous applicants that you can use as a training set for logistic regression
        - for each training example, you have the applicant’s scores on two exams and the admissions decision
        - your task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams
- loading and visualizing the data
    - start by loading the data set
        - the load_dataset() function shown below loads that data into variables X_train and y_train
            - X_train contains exam scores on two exams for a student
            - y_train is the admission decision
                - y_train = 1 if the student was admitted
                - y_train = 0 if the struden was not admitted
            - both X_train and y_train are numpy arrays
    ```python
# load dataset
X_train, y_train = load_data("data/ex2data1.txt")
    ```
    - view the variables
        - good place to start is to just print out each variable and see what it contains
        ```python
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

# First five elements in X_train are:
#  [[34.62365962 78.02469282]
#  [30.28671077 43.89499752]
#  [35.84740877 72.90219803]
#  [60.18259939 86.3085521 ]
#  [79.03273605 75.34437644]]
# Type of X_train: <class 'numpy.ndarray'>
        ```
        ```python
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

# First five elements in y_train are:
#  [0. 0. 0. 1. 1.]
# Type of y_train: <class 'numpy.ndarray'>

        ```
    - check the dimensions of you variables
        - another useful way to get familiar with the data is to view its dimensions
        ```python
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# The shape of X_train is: (100, 2)
# The shape of y_train is: (100,)
# We have m = 100 training examples
        ```
    - visualize your data
        - it is always good to visualize the data if possible
            - the code below displays the data on a 2D plot, where the axes are the two exam scares, and the positive and negative examples are shown with different markers
            - a helper function in the [utils.py](http://utils.py) file to generate this plot was used 
            ```python
# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
            ```
                - your goal is to build a logistic regression model to fit this data
            - with this model, you can then predict if a new student will be admitted based on their scores on the two exams
- Sigmoid function
    - the model is represented as f_w,b(x) = g(w . x + b)
    - where function g is the sigmoid function, which is defined as g(z) = 1/(1 + e^-z)

Exercise 1
- complete the sigmoid function
    - z is not always a single number, but can also be an array of numbers
    - if the input is an array of numbers, we’d like to apply the sigmoid function to each values in the input array
    ```python
# UNQ_C1
# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
          
    ### START CODE HERE ### 
    
    g = 1 / (1 + np.exp(-z))
    
    ### END SOLUTION ###  
    
    return g
    ```
- testing
    ```python
# Note: You can edit this value
value = 0

print (f"sigmoid({value}) = {sigmoid(value)}")

# sigmoid(0) = 0.5
    ```
    ```python
print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)

# sigmoid([ -1, 0, 1, 2]) = [0.26894142 0.5        0.73105858 0.88079708]
# All tests passed!
    ```

Exercise 2
- complete the compute_cost function
    - recall that for logistic regression, the cost function is of the form
        - where
        - m is the number of training examples in the dataset
        - loss(f_w,b(x^i),y^i) is the cost for a single data point, which is
                - f_w,b(x^i) is the model’s prediction, while y^i, which is the actual label
        - f_w,b(x^i) = g(w . x^i + b) where function g is the sigmoid function
            - might be helpful to first calculate an intermediate variable
                        - where n is the number of features, before calculating f_w,b(x^i)  = g(z_w,b(x^i))
    - Note
        - remember that the variables X_train and y_train are not scalar values but matrices of shape (m,n) and (m,1) respectively, where n is the number of features and m is the number of training examples
        - you can use the sigmoid function
    ```python
# UNQ_C2
# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    
    ### START CODE HERE ###
    
    loss_total = 0
    
    for i in range(m):
        f_wb = sigmoid(w@X[i] + b)
        """
        z_wb = 0
        
        for j in range(n):
            z_wb += w[j] * X[i][j]    # without dot product
        z_wb += b
        f_wb = sigmoid(z_wb)
        """
        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        loss_total += loss
        
    total_cost = loss_total/m
    
    ### END CODE HERE ### 

    return total_cost
    ```
- testing
    ```python
m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))
# Cost at initial w and b (zeros): 0.693

    ```
    ```python
# Compute and display cost with non-zero w and b
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))

# UNIT TESTS
compute_cost_test(compute_cost)

# Cost at test w and b (non-zeros): 0.218
# All tests passed!

    ```
- Gradient for logistic regression
    - implement the gradient for logistic regression
    - the gradient descent alg is
        - where parameters b, w_j are all updated simultaneously

Exercise 3
- complete the compute_gradient function to compute
    - m is the number of training examples in the dataset
    - f_w,b(x^i) is the model’s prediction, while y^i is the actual label
    - while this gradient looks identical to the linear regression gradient, the formula is actually different because linear and logistic regression have different definitions of f_w,b(x)
    ```python
# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 
    for i in range(m):
        z_wb = 0
        for j in range(n): 
            z_wb += w[j] * X[i][j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw[j] += (f_wb - y[i]) * X[i][j]
            
    dj_dw /= m
    dj_db /= m
    ### END CODE HERE ###

        
    return dj_db, dj_dw
    ```
    ```python
# Compute and display gradient with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )

# dj_db at initial w and b (zeros):-0.1
# dj_dw at initial w and b (zeros):[-12.00921658929115, -11.262842205513591]

    ```
    ```python
# Compute and display cost and gradient with non-zero w and b
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test w and b:', dj_db)
print('dj_dw at test w and b:', dj_dw.tolist())

# UNIT TESTS    
compute_gradient_test(compute_gradient)

# dj_db at test w and b: -0.5999999999991071
# dj_dw at test w and b: [-44.831353617873795, -44.37384124953978]
# All tests passed!

    ```
- Learning parameters using GD
    - a good way to verify that GD is working correctly is to look at the value of J(w,b) and check that it is decresing with each step
    - assuming you have implemented the gradient and computed the cost correctly, your value of J(w,b) should never increase , and should converge to a steady value by the end of the alg
    ```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing
    ```
    - the following code block below takes a couple minutes to run, especially with a non-vectorized version
    - you can reduce the iterations to test your implementation and iterate faster
    ```python
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)
                                   
# Iteration    0: Cost     0.96   
# Iteration 1000: Cost     0.31   
# Iteration 2000: Cost     0.30   
# Iteration 3000: Cost     0.30   
# Iteration 4000: Cost     0.30   
# Iteration 5000: Cost     0.30   
# Iteration 6000: Cost     0.30   
# Iteration 7000: Cost     0.30   
# Iteration 8000: Cost     0.30   
# Iteration 9000: Cost     0.30   
# Iteration 9999: Cost     0.30
    ```

- Plotting the decision boundary
    - now use the final parameters from gradient descent to plot the linear fit
    ```python
plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
    ```

- Evaluating logistic regression
    - we can evaluate the quality of the parameters we have found by seeing how well the learned model predicts on the training set
    - implement the predict function

Exercise 4
- complete the predict function to produce 1 or 0 predictions given a dataset and a learned prarmeter vector w and b
    - first, compute the prediction from the model f(x^i) = g(w . x^i + b) for every example
    - interpret the output of the model f(x^i) as the probability that y^i = 1 given x^i and parameterized by w
    - to get a final prediction (y^i = 0 or y^i = 1) from the logistic regression model, you can use the following heuristic
        - if f(x^i) ≥ 0.5, predict y^i = 1
        - if f(x^i) \< 0.5, predict y^i = 0
    ```python
# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += w[j] * X[i][j]
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = f_wb >= 0.5
        
    ### END CODE HERE ### 
    return p
    ```
- testing
    ```python
# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

# UNIT TESTS        
predict_test(predict)

# Output of predict: shape (4,), value [0. 1. 1. 1.]
# All tests passed!
    ```
- compute accuracy
    ```python
#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

# Train Accuracy: 92.000000
    ```

Regularized Logistic Regression
- predict whether microchips from a grabrication plant passes quality assurance (QA)
- each microchip goes through various tests to ensure it is function correctly
- Problem Statement
    - suppose you are the product manager o the factory and you have the test results for some microchips on two different tests
        - from these two test, you would like to determine whether microchips should be accepted ot rejected
        - to help make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model
- Loading and visualizing the data
    - start by loading the dataset for this task and visualizing it
        - the load_dataset() function loads the data variables X_train and y_train
            - X_train contains the test results for the microchips from two test
            - y_train contains the results of the QA
                - y+train = 1 if the microchip was accepted
                - y_train = 0 if the microchip was rejected
            - both X_train and y_train are numpy arrays
        ```python
# load dataset
X_train, y_train = load_data("data/ex2data2.txt")
        ```
    - view the variables
        - print the first five values of X_train and y_train and the type of the variables
        ```python
# print X_train
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))

# print y_train
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

# X_train: [[ 0.051267  0.69956 ]
#  [-0.092742  0.68494 ]
#  [-0.21371   0.69225 ]
#  [-0.375     0.50219 ]
#  [-0.51325   0.46564 ]]
# Type of X_train: <class 'numpy.ndarray'>
# y_train: [1. 1. 1. 1. 1.]
# Type of y_train: <class 'numpy.ndarray'>
        ```
    - check the dimensions of your variables
        ```python
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# The shape of X_train is: (118, 2)
# The shape of y_train is: (118,)
# We have m = 118 training examples
        ```
    - visualize the data
        - the helper function plot_data is used to generate the figure
        ```python
# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()
        ```
            - the dataset cannot be separated into positive and negative examples by a straight-line throught the plot
        - a straight forward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary
- Feature mapping
    - one way to fit the data better is to create more features from each data point
    - in the function map_feature, more features will be mapped into all polynomial terms of x_1 and x_2 up to the sixth power
        - as a result of this mapping, our vector of two features (the scores on two QA test) has been transformed into a 27D vector
        - a logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will be nonlinear when drawn in our 2D plot
    ```python
print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

# Original shape of data: (118, 2)
# Shape after feature mapping: (118, 27)
    ```
    ```python
print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])

# X_train[0]: [0.051267 0.69956 ]
# mapped X_train[0]: [5.12670000e-02 6.99560000e-01 2.62830529e-03 3.58643425e-02
#  4.89384194e-01 1.34745327e-04 1.83865725e-03 2.50892595e-02
#  3.42353606e-01 6.90798869e-06 9.42624411e-05 1.28625106e-03
#  1.75514423e-02 2.39496889e-01 3.54151856e-07 4.83255257e-06
#  6.59422333e-05 8.99809795e-04 1.22782870e-02 1.67542444e-01
#  1.81563032e-08 2.47750473e-07 3.38066048e-06 4.61305487e-05
#  6.29470940e-04 8.58939846e-03 1.17205992e-01]
    ```
    - while the feature mapping allows for a more expressive classifier, it is also more susceptible to overfitting
    - implement regularized logistic regression to fit the data and also see how reg can help combat the overfitting problem
- Cost function for regularized logsitic regression
    - implement the cost function for reg logistic regression
        - for logistic regression , the cost function is of the form

Exercise 5
- complete the compute_cost_reg function below to calculate the following term for each element in w
    ```python
# UNQ_C5
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost 
    """

    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = 0.
    
    ### START CODE HERE ###
    for j in range(n):
        reg_cost += w[j]**2
        
    reg_cost = reg_cost * lambda_ / (2 * m)
    ### END CODE HERE ### 
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost
    ```
- testing
    ```python
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# UNIT TEST    
compute_cost_reg_test(compute_cost_reg)

# Regularized cost : 0.6618252552483948
# All tests passed!
    ```

- Gradient for reg logistic regression
    - implement the gradient for reg logistic regression
    - the gradient of the regularized cost function has two components
    - the first, w.r.t b is a scalar
    - the other is a vector with the same shape as the parameters w, where the jth element is defined:
    
Exercise 6
- complete the compute_gradient_reg function below to modify the code below to calculate the (alpha*w_j)/m term
    ```python
# UNQ_C6
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for logistic regression with regularization
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    ### START CODE HERE ###     
    for j in range(n):
        dj_dw[j] += (lambda_ / m * w[j])
    ### END CODE HERE ###         
        
    return dj_db, dj_dw
    ```
- testing
    ```python
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1) 
initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
initial_b = 0.5
 
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

# UNIT TESTS    
compute_gradient_reg_test(compute_gradient_reg)

# dj_db: 0.07138288792343662
# First few elements of regularized dj_dw:
# [-0.010386028450548701, 0.011409852883280122, 0.0536273463274574, 0.0031402782673134655]
# All tests passed!
    ```

- Learning parameters using GD
    - use GD function to learn the optimal parameters w,b
    - note: the code block will take a while to run, especially with a non-vectorized version (it is vectorized i think because of @)
    ```python
# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01    

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)
                                    
# Iteration    0: Cost     0.72   
# Iteration 1000: Cost     0.59   
# Iteration 2000: Cost     0.56   
# Iteration 3000: Cost     0.53   
# Iteration 4000: Cost     0.51   
# Iteration 5000: Cost     0.50   
# Iteration 6000: Cost     0.48   
# Iteration 7000: Cost     0.47   
# Iteration 8000: Cost     0.46   
# Iteration 9000: Cost     0.45   
# Iteration 9999: Cost     0.45
    ```

- Plotting the decision boundary
    - use plot_decision_boundary function which plots the (non-linear) decision boundary that separates the positive and negative examples
        - the non-linear decision boundary plotted by computing the classifier’s prediction son an evenly spaced grid and then drew a contour plot of where the predictions change from y = 0 to y = 1
        - after learning the parameters w,b the next step is to plot a decision boundary
        
- Evaluating reg logistic regression model
    - use the predict function to calculate the accuracy of the re logistic regression model on the training set
    ```python
#Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

# Train Accuracy: 82.203390
    ```
