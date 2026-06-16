- AGI = artificial general intelligence
- Machine Learning - Field of study that gives computers the ability to learn without being explicitly programmed." - Arthur Samuel (1959)

Machine learning algorithms
- supervised learning - used most in real-world application
- unsupervised learning
- recommender systems
- reinforcement learning

Supervised learning
- algorithms that learn input to output maps

Regression: Housing price prediction
- later in the course → how to decide to fit line, curve, or something more complex
- the task of the learning algorithm is to produce more of these right answers

Classification algorithm
- breast cancer detection → either breast is benign or malignant
- different from regression because classification tries to predict only a small number of outputs or categories
- class and category are interchangeable
- can be non-numeric

Two or more inputs
- machine would learn to create a boundary between the inputs

Unsupervised learning
- inputs are not given with any labels y
- find something interesting or a pattern in unlabeled data
- *clustering* algorithms
- *anomaly detection* - find unusual data points
- *dimensionality reduction* - compress data using fewer numbers

Jupyter Notebooks
- run code one line at a time
- does vscode have jupyter notebook?
- markdown cell → text editing
- code cell → for code
- shift → enter will convert markdown back to normal or run the code cell

### Optional Lab: Brief Introduction to Python and Jupyter Notebooks
- two types of cells, markdown cells and code cells
- f strings allow you to embed variables
  - print (f"something {var}")

### Linear Regression with One Variable

House sizes and prices
- you can build a linear regression model that will fit a straight line to the data

Regression → many possible outputs, range
Classification → distinct outputs

Terminology
- Training set → data used to train the model
  - x = "input" variable, feature
  - y = "output" variable, "target" variable
  - m = number of training examples → (x, y), single training example
  - (x^(i), y^(i)) = ith training example
  - x^(i) is not exponentiation

Training set → learning algorithm → function (f, hypothesis)
- the job of f is to take x as input and produce a y-hat
- f is called the model
- y-hat is the prediction → the estimated y
- with housing, the model f outputs an estimated price of the house from an input size

Key questions: how to represent f?
- f_(w,b)(x) = wx + b
  - w and b are numbers and those values determine the prediction y-hat
- can also be written of f(x)

Why linear?
- line is a foundation for non-linear complex models

In the example of one house input of size, the model would be linear regression with one variable, or single feature x. Also know as univariate linear regression.

### Optional Lab: Model Representation

Notation
- a → scalar, non bold (i believe scalar is just a number)
- **a** → vector, bold (and vector is a direction, like (x, y))
- **x** → training example feature values → x_train in python
- **y** → training example targets → y_train in python
- x^i, y^i → i_th training example → x_i, y_i
- m → number of training examples → m
- w → parameter: weight → w
- b → parameter: bias → b
- f_(w,b)(x^i) → the result of the model evaluation at x^i parameterized by w, b → f_wb

Tools
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data

What is plt.style.use('./deeplearning.mplstyle')
- Matplotlib style sheet I think, found in the notebook files

x_train is the input variable (size in 1000 square feet)
- x_train = np.array([1.0, 2.0]) → x_train = [1. 2.] (1. mean 1.0 I think)

y_train is the target (price in 1000s of dollars)
- y_train = np.array([300.0, 500.0]) → y_train = [300. 500.]

Number of training examples m
- numpy array have a .shape parameter
- x_train.shape: (2,) → returns a python tuple with an entry for each dimension
- x_train.shape[0]: 2 → is the length of the array and number of examples
- what is a python tuple?
  - A Python tuple is an ordered, immutable collection of elements. It is similar to a list, but with the key difference that once a tuple is created, its elements cannot be modified, added, or removed.
- len() function returns count of items in passed object

Training example x_i, y_i
- (x^i, y^i) to denote the ith training example
- python is zero indexed, (x^0, y^0) is (1.0, 300.0) and (x^1, y^1) is (2,0, 500.0)

Plotting the data
- scatter() function in the matplotlib library
- the function arguments marker and c show the points as red crosses (default is blue dots)
- plt. to access matplotlib functions
- plt.title for title
- plt.ylabel and plt.xlabel for x and y labels
- plt.show() to display

Start with seemingly random w and b
- w = 100
- b = 100

Compute the value of f_(w,b)(x^i) for the two data points
- x^0 → f_wb = w * x[0] + b
- x^1 → f_wb = w * x[1] + b
- (ndarray (m,)) describes a Numpy n-dimensional array of shape (m,). (scalar) describes an argument without dimensions, just a magnitude
- np.zero(n) will return a 1D numpy array with n entries

```python
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb
```

- plt.plot(...) → used to create line plots
- plt.scatter(...) for creating scatter plots

Prediction
- to predict the price of a house with 1200 sqft
- x = 1.2, w = 200, b = 100 (no reason, just fit best after fiddling)
- hardcode the prediction → cost_1200sqft = w * x_i + b
- cost_1200sqft:.0f = 340 instead of 340.0

Cost Function
- tells us how well the model is doing

Parameters
- w and b are the parameters
- variables you can adjust during training to improve the model
- coefficients or weights
- w is the slope
- b is the intercept with the y-axis

Find w, b:
- y_hat^i is close to y^i for all (x^i, y^i)

Cost function
- y_hat - y → error
- the error will be squared (why?)
- sum them up to the number of training examples
- more examples → cost function gets bigger
- instead, by convention, compute average square error by dividing by m
- actually divide by 2m, calculations make them neater
- known as squared error cost function
- most widely used for regression problems

Cost Function Intuition
- model: f_w,b(x) = wx + b
- parameters: w, b
- cost function: J(w.b) = (1/2m)sum^(m)_(i=1)(f_w,b(x^i) - y(i))^2
  - measures the difference between the model's predictions and the actual true values for y
- goal: minimize J(w,b)

Simplified
- f_w(x) = wx
- J(w) = (1/2m)sum^(m)_(i=1)(f_w(x^i) - y^i)^2
- minimize J(w)
- when m = 1, the cost function is equal to 0
- when m = 0.5, the cost function is equal to ~0.58
- when m = 0, the cost function is equal to ~2.3

Choose w to minimize J(w)
- goal of linear regression: minimize J(w)
- general case: minimize J(w,b)

Cost function shape with w and b
- the lines on the top left graph are the dots on the top right and bottom

### Optional Lab: Cost Function

- %matplotlib widget: a Jupyter Notebook magic command that enables interactive Matplotlib plots within a widget-based window
- matplotlib.pyplot: is a collection of functions that make matplotlib work like MATLAB
- MATLAB: a graphical programming language used for creating simulations and controlling hardware
- from lab_utils_uni import ... : imports functions from file

Computing Cost
- Note, summation ranges are typically from 1 to m, while code will be from 0 to m-1

Cost in code below is calculated by looping over each example, in each loop:
- f_wb, a prediction is calculated
- the difference between the target and the prediction is calculated and squared
- then added to the total cost

```python
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost
```

- because the difference between the target and prediction is squared in the cost equation, the cost increases rapidly when w is either too large or too small
- from the interwebs: the loss function measures the error for a single data point, while a cost function aggregates the loss over an entire dataset or batch
- the fact that the cost function squares the loss ensures that the 'error surface' is convex like a soup bowl
- it will always have a minimum that can be reached by following the gradient in all dimensions

Gradient Descent
- Have some function J(w,b)
- for linear regression or any function
- Want min_(w,b)J(w,b)
- works for cost function of many variables

Outline:
- Start with some w, b → set usually to w = 0, b = 0
- keep changing w, b to reduce J(w, b)
- Until we settle at or near a minimum
- Some functions have more than one possible minimum

Direction of steepest descent
- local minima will be found if you start at different points on the curve

Gradient descent algorithm
- assignment → related to code
- truth assertion → related to math
- in code, truth assertion would be written as ==
- alpha = learning rate → small number, how big of a step
- derivative of the cost function → which direction

Correct: Simultaneous update

Gradient Descent Intuition
- the 'derivative' gives the slope of the line tangent the point on the curve wrt a given w
- the learning rate is applied and w moves towards a downward slope

Learning rate
- if alpha is too small
  - too slow
- if alpha is too large
  - bounces past the minima back and forth → thrashing
  - fail to converge, diverge

If you're at a local minimum, gradient descent leaves w unchanged
- can reach local minimum with fixed learning rate
- Near a local minimum, GD will automatically get smaller

Gradient Descent for Linear Regression
- convex function → bowl shaped → no local minima except the single global minima

Batch Gradient Descent
- 'batch': each step of gradient descent uses all the training examples
- other versions that use subsets

### Optional Lab: Gradient Descent for Linear Regression

Conventions
- naming of python variables containing partial derivatives follows this pattern → dj_db
- wrt is With Respect To, as in partial derivate of J(wb) With Respect To b

Gradient Compute:

```python
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db
```

Cost Function:

```python
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i],w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2*m)
    return cost
```

Gradient Descent:

```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        if i<100000:
            J_history.append(cost_function(x, y, w , b))
            p_history.append([w,b])
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history
```

Cost versus iterations of gradient descent
- a plot of cost versus iterations is a useful measure of progress in gradient descent
- cost should always decrease in successful runs
- the change in cost is so rapid initially, it is useful to plot the initial decent on a different scale than the final descent

Increased Learning Rate
