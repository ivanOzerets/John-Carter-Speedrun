Welcome

- neural networks
- decision trees
- practical advice
- GPU vs more data?

Advanced learning algorithms

- Week 1-2: Neural Networks
    - inference (prediction) → using someone else’s neural network to make predictions
    - training
- Week 3: Practical advice for building machine learning systems
- Week 4: Decision Trees

Neural Network

- origins: algs that try to mimic the brain
- used in 80s and early 90s
- fell out of favor in late 90s
- resurgence from around 05
- speech recognition → image → text (NLP) → ChatGPT (i’m assuming)
- simplified mathematical model of a neuron

Why now?

- traditional AI → linear regression and logistic regression

Demand Prediction

- neuron can be thought of as a tiny computer whose job is to input one or a few numbers and output one number of a few numbers
- sigmoid function is mapped to data and f(x) is the activation a → output (for example)
- activation values → output of a layer
- input layers don’t have to match size of layer neurons
- each neuron could be responsible for which product features are important

- a lot of work if you were to assign which each neuron should take which features as inputs
- in practice, each neuron will have access to every input feature
- output layer,  input layer, ‘hidden’ layer

- another way of thinking about neural networks is to imagine the output linear regression ‘neuron’ (pink) is getting as input the three activation values so it’s really the same as before except the features can be though of as ‘better’ that are more of a predictor of the output
- just logistic regression but can learn its own features

Multiple hidden layers

- how many hidden layers and how many neurons per hidden layer?
- neural network architecture
- multi-layer perceptron

Face recognition

- brightness intensity values go from 0 - 255
- face image is a matrix of pixels
- the pixel brightness as a list are the input data

- the earliest layers of a layer, you will find that the neurons are looking for lines or edges
- the next layer will learn to group together short lines and edge segments
- the next layer could aggregate large parts of faces to detect presence of larger face shapes
- can learn there feature detectors
- neuron layers progressively scale their region of focus

Neural network layer

- neurons have parameters w_1 and b_1 and outputs a_1
- a_1 = g(z) = g(w . x + b)

- layers have number order left to right
- a^[1] is notion to denote output of layer 1
- second layer / output layer

- if binary prediction → threshold ≥ 0.5
- y_hat = 1 if a[2] is ≥ 0.5

More complex neural networks

- within each layer, the parameters w and b and output a have that layer’s number in the square brackets

- arbitrary activation vector

- sigmoid is the “activation function”
- input vector x is activation a[0]

Handwritten digit recognition

- between 0 and 1
- eight by eight image of 266 pixel values

- a[3] = f(x)
- x / a[0] → a[1] → a[2] → a[3] → forward propagation
- hidden units decreasing per layer is typical

Coffee roasting

- the same alg can be applied to so many different applications

- inference → get a NN to tell us whether the input values create good coffee
- Dense → another name for the layers, current layer type so far

Building the model using TensorFlow

- PyTorch is the other library for handling NNs
- building the model

    ```python
    x = np.array([[200.0, 17.0]])
    layer_1 = Dense(units=3, acivation='sigmoid')
    a1 = layer_1(x)
    ```

    ```python
    layer_2 = Dense(units=1, activation='sigmoid')
    a2 = layer_2(a1)
    ```

    ```python
    if a2 >= 0.5:
    	yhat = 1
    else:
    	yhat = 0
    ```

Feature vectors

- when double square bracket [[
- note about numpy arrays
    - the inner brackets are always the first row
    - [[200, 17]] → [200, 17] → 1 x 2 → row vector
    - [[200], [17]] → [200 17] (stacked on top of each other) → 2 x 1 → column vector
    - difference between double square and single is 1D vs 2D vector
    - convention in TensorFlow is to represent data in matrices
- Tensor is a data type → matrix
- a.numpy() will return a Tensor matrix/array into a numpy array
- TensorFlow can take numpy arrays as input, it will just convert to Tensors

Building a neural network architecture

- TensorFlow can grab layers and make them a ‘model’ to execute sequential forward propagation steps
- model = Sequential([layer_1, layer{2])
- model.compile(…) → model.fit(x,y) → model.predict(x_new)
- usually just construct under model rather than having an explicit assignment to layer variables

    This code will define a NN with how many layers? 

    - [x]  4
    - [ ]  3
    - [ ]  5
    - [ ]  25
- How do you define the second layer of a NN that has 4 neurons and a sigmoid activation?
    - [x]  Dense(units=4, activation=’sigmoid’)
    - [ ]  Dense(layer=2, units=4, activation=’sigmoid’)
    - [ ]  Dense(units=[4], activation=[’sigmoid’])
    - [ ]  Dense(units=4)
- If the input features are temp and duration, how do you write the code for the feature vector x
    - [ ]  x = np.array([[200.0],[17.0]])
    - [ ]  x = np.array([[200.0 + 17.0]])
    - [x]  x = np.array([[200.0, 17.0]])
    - [ ]  x = np.array([[’200.0’,’17.0’]])

Forward prop (coffee roasting model)

- w^[2]_1 will be denoted as w2_1

Forward prop in numpy / manual forward prop

- function for the dense layer
- input as activation from previous layer, the weight matrix, and bias array
- shape() is used to pull out number of hidden units, or columns or rows
- W[:,j] → all rows (:), just j column

- capital W refers to a matrix
- lower case is for vector and scalar

AI

- can be broken into two categories
- ANI (artificial narrow intelligence) → smart speaker, self-driving car, web search / applied to specific applications
- AGI (artificial general intelligence) → do anything a human can do
- we have almost no idea how the brain works

The “one learning algorithm” hypothesis

- one piece of biological brain tissue can do a wide range of tasks
- maybe a lot of intelligence is due to one or a handful of learning algs
- auditory cortex learns to see once the audio connection is cut and fed images

NN can be vectorized / vectorized implementations

- np.matmul() or @ doesn’t allow scalar inputs and handles stacked matrices by broadcasting their product → matrix x matrix multiplication
- np.dot() includes scalar multiplication and treats higher-dimensional arrays as a sum of dot products → performs a more general product

Dot products

- multiply element-wise and add
- transpose is switching vector orientation
- z = a . w = a^T x w
- the operation of multiplying two vectors [m,n]*[n,m] and producing [m,m] is the same as take the dot product of the two even if the dimensions don’t match

Vector matrix multiplication

- same rules as before

Matrix matrix multiplication

- just a set of vectors stacked together
- think of matrices as columns, transpose as the rows

Matrix multiplication rules

- inner product is the same as dot product
- A^T x W → first matrix (transposed) is responsible for the rows, second is responsible for columns

- can only take dot products of vectors that are the same length
- so inner numbers for dimensions need to match
- output will have the same dimensions as the outer numbers

Vectorization in code

- A.T is the transpose function
- AT @ W is alternative for matmul()

- a_in is the same as AT (A transposed)
- convention that inputs for x are laid out in rows → why you need to transpose