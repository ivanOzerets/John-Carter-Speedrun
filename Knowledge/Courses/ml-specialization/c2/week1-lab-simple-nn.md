- build a small NN using numpy
- same as the TensorFlow one from last lab

DataSet

- same data set
- plot the data

    ```python
    X,Y = load_coffee_data();
    print(X.shape, Y.shape)

    # (200, 2) (200, 1)
    ```

Normalize Data

- normalize the data, same as last lab

```python
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

'''
Temperature Max, Min pre normalization: 284.99, 151.32
Duration    Max, Min pre normalization: 15.45, 11.51
Temperature Max, Min post normalization: 1.66, -1.69
Duration    Max, Min post normalization: 1.79, -1.70
'''
```

Numpy Model (Forward Prop in numpy)

- two layers with sigmooid activations
- it is possible to build your own dense layer using numpy
- in the first lab, a neuron in numpy and in tensorflow was constructed
- a layer simply contains multiple nearon/units
- one can utilize a for loop to vistt each unit (j) in the layer and perform the dot product of the weights for that unit (W[:,j]) and sum the bias for the unit (b[j]) to form z
- the activation function g(z) can then be applied to that result
- define the activation function g()
- use the sigmoid() function (predefined)

    ```python
    # Define the activation function
    g = sigmoid
    ```

- define the my_dense() function which computes the activatios of a dense layer

    ```python
    def my_dense(a_in, W, b):
        """
        Computes dense layer
        Args:
          a_in (ndarray (n, )) : Data, 1 example 
          W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
          b    (ndarray (j, )) : bias vector, j units  
        Returns
          a_out (ndarray (j,))  : j units|
        """
        units = W.shape[1]
        a_out = np.zeros(units)
        for j in range(units):               
            w = W[:,j]                                    
            z = np.dot(w, a_in) + b[j]         
            a_out[j] = g(z)               
        return(a_out)
    ```

- Note: the function can be implemented to accept g as an additional parameter
- in this notebook, there is only use of one activation function, so it’s okay to make it constant and define it outside the function
- build a two-layer neural network utilizing the my_dense subroutine

    ```python
    def my_sequential(x, W1, b1, W2, b2):
        a1 = my_dense(x,  W1, b1)
        a2 = my_dense(a1, W2, b2)
        return(a2)
    ```

- copy trained weights and biases from the previous lab

    ```python
    W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
    b1_tmp = np.array( [-9.82, -9.28,  0.96] )
    W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
    b2_tmp = np.array( [15.41] )
    ```

Predications

- make predication with the trained model
- the output of the model is a probability
- apply the probability to a threshold → 0.5
- start by writing a routine similar to TensorFlow’s model.predict()
- take a matrix X with all m examples in the rows and make a prediction by running the model

    ```python
    def my_predict(X, W1, b1, W2, b2):
        m = X.shape[0]
        p = np.zeros((m,1))
        for i in range(m):
            p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
        return(p)
    ```

- try routine on two examples

    ```python
    X_tst = np.array([
        [200,13.9],  # postive example
        [200,17]])   # negative example
    X_tstn = norm_l(X_tst)  # remember to normalize
    predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
    ```

- convert the probabilities to a decision, we apply a threshold

    ```python
    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    '''
    decisions = 
    [[1.]
     [0.]]
    '''
    ```

- or more succinctly

    ```python
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")

    '''
    decisions = 
    [[1.]
     [0.]]
    '''
    ```

Network function

- the graph shows the operation of the whole network and is identical to the TensorFlow result from the previous lab
- the left graph is the raw output of the final layer represented by the blue shading
- the right graph is the output of the network after a decision threshold
- the Xs and Ox correspond to decisions made by the network

    ```python
    netf= lambda x : my_predict(norm_l(x),W1_tmp, b1_tmp, W2_tmp, b2_tmp)
    plt_network(X,Y,netf) # don't understand this lamda thing yet
    ```