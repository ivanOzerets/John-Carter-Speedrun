- recurrent NN are very effective for NLP and other sequence tasks because they have memory
- they can read inputs x<t> one at a time, and remember some contextual information through the hidden layer activations that get passed from one time step to the next
- this allows a unidirectional RNN to take information from the past to process later inptus
- a bidirectional RNN can take context from both the past an the future

- Notation
    - superscript [l] denotes an object associated with the lth layer
    - superscript (i) denotes an object associated with the ith example
    - superscript <t> denotes an object at the tth time step
    - subscript i denotes the ith entry of a vector
    - a^(2)[3]<4>_5 denotes the activation of the 2nd training example, third layer, fourth timestep, and fifth entry in the vector

Packages

```python
import numpy as np
from rnn_utils import *
from public_tests import *
```

Forward Prop for the Basic Recurrent NN

- Tx = Ty

- Dimensions of input x
    - input with nx number of units
        - for a single time step of a single input example, x(i)>t? is a one dimensional input vector
        - using language as an example, a language with a 5k work vocabulary could be one-hot encoded into a vector that has 5k units
        - so x(i)<t> would have the shape (5k,)
        - the notation nx is used to denote the number of units in a single time step of a single training example
    - time steps of size Tx
        - a recurrent NN has multiple time steps, which are indexed with t
        - Tx will denote the number of timesteps in the longest sequence
    - batches of size m
        - mini-batches of 20 training examples
        - to benefit from vectorization, stack 20 columns of x(i) examples
        - for example, (5k, 20, 10)
        - use m to denote the number of training examples
        - the shape of a mini-batch is (nx, m, Tx)
    - 3D tensor of shape (nx, m, Tx)
        - the 3D tensor x represents the input x that is fed into the RNN
    - taking a 2D slice for each time step: x<t>
        - at each time step, use a mini-batch of training examples
        - for each time step t, use a 2D slice of shape (nx, m)
        - this 2D slice is referred to as x<t>
- Definition of hidden state a
    - the activation a<t> that is passed to the RNN from one time step to another is called a hidden state
- Dimensions of hidden state a
    - similar to the input tensor x, the hidden state for a single training example is a vector of length na
    - including mini-batches of m training examples, the shape of a mini-batch is (na, m)
    - when including the time step dimension, the shape of the hidden state is (na, m, Tx)
    - loop through he time steps with index t, and work with a 2D slice of the 3D tensor
    - this 2D slice is referred to as a<t>
    - the shape of the 2D slice is (na, m)
- Dimensions of prediction y_hat
    - similar to the inputs and hidden states, y_hat is a 3D tensor of shape (ny, m, Ty)
        - n_y: number of units in the vector representing the prediction
        - m: number of examples in a mini-batch
        - Ty: number of time steps in the prediction
    - for a single time step t, a 2D slice y_hat<t> has shape (ny, m)
- Steps
    - implement the calculations needed for one time step of the RNN
    - implement a loop over Tx time steps in order to process all the inputs, one at a time
- RNN Cell
    - think of the recurrent NN as the repeated use of a single cell
    - first, implement the computations for a single time step

    - RNN cell vs RNN_cell_forward
        - an RNN cell outputs the hidden state a<t>
            - RNN cell is the solid box above
        - the function, rnn_cell_forward, also calculates the prediction y_hat<t>
            - shown is the dashed box
    - Exercise 1 - rnn_cell_forward
        - compute the hidden state with tanh activation
        - using the new hidden state a<t>, compute the prediction
        - store (a<t>, a<t-1>, x<t>, params) in a cache
        - return a<t>, y_hat<t> and cache

        ```python
        # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: rnn_cell_forward

        def rnn_cell_forward(xt, a_prev, parameters):
            """
            Implements a single forward step of the RNN-cell as described in Figure (2)

            Arguments:
            xt -- your input data at timestep "t", numpy array of shape (n_x, m).
            a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
            parameters -- python dictionary containing:
                                Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                                Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                                Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                                ba --  Bias, numpy array of shape (n_a, 1)
                                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
            Returns:
            a_next -- next hidden state, of shape (n_a, m)
            yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
            cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
            """

            # Retrieve parameters from "parameters"
            Wax = parameters["Wax"]
            Waa = parameters["Waa"]
            Wya = parameters["Wya"]
            ba = parameters["ba"]
            by = parameters["by"]

            ### START CODE HERE ### (≈2 lines)
            # compute next activation state using the formula given above
            a_next = np.tanh(Wax @ xt + Waa @ a_prev + ba)
            # compute output of the current cell using the formula given above
            yt_pred = softmax(Wya @ a_next + by)
            ### END CODE HERE ###

            # store values you need for backward propagation in cache
            cache = (a_next, a_prev, xt, parameters)

            return a_next, yt_pred, cache
        ```

- RNN Forward Pass
    - a RNN is a repetition of the RNN cell
        - if the input sequence of data is ten time steps long, then reuse the RNN cell ten times
    - each cell takes two inputs at each time step
        - a<t-1>: the hidden state from the previous cell
        - x<t>: the current time step’s input data
    - it has two outputs at each time step
        - a hidden state a<t>
        - a prediction y<t>
    - the weights and biases (Waa, Wax, ba, Way, by) are reused each time step
        - maintained between calls to rnn_cell_forward in the params dictionary

    - Exercise 2 - rnn_forward
        - create a 3D array of zeros, a of shape (na, m, Tx) that will store all the hidden states computed by the RNN
        - create a 3D array of zeros, y_hat of shape (ny, m, Tx) that will store the predictions
        - initialize the 2D hidden state a_next by setting it equal to the initial hidden state a0
        - at each time step t
            - get x<t>, which is a 2D slice of x for a single time step t
                - x<t> has shape (nx, m)
                - x has shape (nx, m, Tx)
            - update the 2D hidden state a<t>, the prediction y_hat<t> and the cache by running rnn_cell_forward
                - a<t> has shape (na, m)
            - store the 2D state in the 3D tensor a, at the ith position
                - a has shape (na, m, Tx)
            - store the 2D y_hat<t> prediction in the 3D tensor y_hatpred at the ith position
                - y_hat<t> has shape (ny, m)
                - y_hat has shape (ny, m, Tx)
            - append the cache to the list of caches
        - return the 3D tensor a and y_hat, as well as the list of caches

        ```python
        # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: rnn_forward

        def rnn_forward(x, a0, parameters):
            """
            Implement the forward propagation of the recurrent neural network described in Figure (3).

            Arguments:
            x -- Input data for every time-step, of shape (n_x, m, T_x).
            a0 -- Initial hidden state, of shape (n_a, m)
            parameters -- python dictionary containing:
                                Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                                Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                                Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                                ba --  Bias numpy array of shape (n_a, 1)
                                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

            Returns:
            a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
            y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
            caches -- tuple of values needed for the backward pass, contains (list of caches, x)
            """

            # Initialize "caches" which will contain the list of all caches
            caches = []

            # Retrieve dimensions from shapes of x and parameters["Wya"]
            n_x, m, T_x = x.shape
            n_y, n_a = parameters["Wya"].shape

            ### START CODE HERE ###

            # initialize "a" and "y_pred" with zeros (≈2 lines)
            a = np.zeros((n_a, m, T_x))
            y_pred = np.zeros((n_y, m, T_x))

            # Initialize a_next (≈1 line)
            a_next = a0

            # loop over all time-steps
            for t in range(T_x):
                # Update next hidden state, compute the prediction, get the cache (≈1 line)
                a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
                # Save the value of the new "next" hidden state in a (≈1 line)
                a[:,:,t] = a_next
                # Save the value of the prediction in y (≈1 line)
                y_pred[:,:,t] = yt_pred
                # Append "cache" to "caches" (≈1 line)
                caches.append(cache)
            ### END CODE HERE ###

            # store values needed for backward propagation in cache
            caches = (caches, x)

            return a, y_pred, caches
        ```

Long Short-Term Memory (LSTM) Network

- operations of an LSTM cell

- Overview of gates and states
    - forget gate Gamma_f
        - assume reading words in a piece of text, and plan to use an LSTM to keep track of grammatical structures such as whether the subject is singular or plural
        - if the subject changes its state (from a singular word to a plural word), the memory of the previous state becomes outdates, so the outdated state will be forgotten
        - the forget gate is a tensor containing values between 0 and 1
            - if a unit in the forget gate has a value close to 0, the LSTM will forget the stored state in the corresponding unit of the previous cell state
            - if a unit in the forget gate has a value close to 1, the LSTM will mostly remember the corresponding value in the stored state

        - Wf contains weights that govern the forget gate’s behavior
        - the previous time step’s hidden state [a<t-1> and current time step’s input x<t>] are concatenated together and multiplied by Wf
        - a sigmoid function is used to make each of the gate tensor’s values Gamma<t>f range from 0 to 1
        - the forget gate Gamma<t>f has the same dimensions as the previous cell state c<t-1>
        - this means that the two can be multiplied together, element-wise
        - if a single value in Gamma<t>f is 0 or close to 0, then the product is close to 0
            - this keeps the information stored in the corresponding unit in c<t-1> from being remembered for the next time step
        - if one value is close to 1, the product is close to the original value in the previous cell state
            - the LSTM will keep the information from the corresponding unit of c<t-1>, to be used in the next time step
    - Candidate value ~c<t>
        - the candidate value is a tensor containing information from the current time step that may be stored in the current cell state c<t>
        - the parts of the candidate value that get passed on depend on the update gate
        - the candidate value is a tensor containing values that range from -1 to 1
        - the tilde is used to differentiate the candidate from the cell state

        - the tanh function produces the value from -1 to 1
    - Update gate Gamma_i
        - the update gate is used to decide what aspects of the candidate ~c<t> to add to the cell state c<t>
        - the update gate decides what parts of a candidate tensor ~c<t> to be passed onto the hidden state c<t>
        - the update gate is a tensor containing values between 0 and 1
            - when a unit in the update gate is close to 1, it allows the value of the candidate ~c<t> to be passed onto the hidden state c<t>
            - when a unit in the update gate is close to 0, it prevents the corresponding value in the candidate from being passed onto the hidden state
        - the subscript i is used and not u

        - the sigmoid produces values between 0 and 1
        - the update gate is multiplied element-wise with the candidate, and this product is used to determine the cell state c<t>
    - Cell state c<t>
        - the cell state is the memory that gets passed onto future time steps
        - the new cell state c<t> is a combination of the previous cell state and the candidate value

        - the previous cell state c<t-1> is adjusted by the forget gate Gamma<t>f
        - the candidate value ~c<t> is adjusted by the update gate Gamma<t>i
    - Output gate Gamma_o
        - the output gate decides what gets sent as the prediction of the time step
        - the output gate is like the other gates, in that it contains values that range from 0 to 1

        - the output gate is determined by the previous hidden state a<t01> and the current input x<t>
        - the sigmoid make the gate range form 0 to 1
    - Hidden state a<t>
        - the hidden state gets passed to the LSMT cell’s next time step
        - it is used to determine the three gates of the next time step
        - the hidden state is also used for the prediction y<t>

        - the hidden state a<t> is determined by the cell state c<t> in combination with the output gate Gamma_o
        - the cell state is passed through the tanh function to rescale values between 01 and 1
        - the output gate acts like a mask that either preserves the values of tanh(c<t>) or keeps those values from being included in the hidden state a<t>
    - Prediction y<t>pred
        - the prediction in this use case is a classification, so use a softmax

- LSTM Cell
    - Exercise 3 - lstm_cell_forward
        - concatenate the hidden state a<t-1> and input x<t> into a single matrix

        - compute all formulas for the gates, hidden state, and cell state
        - compute the prediction y<t>

        ```python
        # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: lstm_cell_forward

        def lstm_cell_forward(xt, a_prev, c_prev, parameters):
            """
            Implement a single forward step of the LSTM-cell as described in Figure (4)

            Arguments:
            xt -- your input data at timestep "t", numpy array of shape (n_x, m).
            a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
            c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
            parameters -- python dictionary containing:
                                Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                                bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                                Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                                bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                                Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                                bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                                Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                                bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                                Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

            Returns:
            a_next -- next hidden state, of shape (n_a, m)
            c_next -- next memory state, of shape (n_a, m)
            yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
            cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

            Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
                  c stands for the cell state (memory)
            """

            # Retrieve parameters from "parameters"
            Wf = parameters["Wf"] # forget gate weight
            bf = parameters["bf"]
            Wi = parameters["Wi"] # update gate weight (notice the variable name)
            bi = parameters["bi"] # (notice the variable name)
            Wc = parameters["Wc"] # candidate value weight
            bc = parameters["bc"]
            Wo = parameters["Wo"] # output gate weight
            bo = parameters["bo"]
            Wy = parameters["Wy"] # prediction weight
            by = parameters["by"]

            # Retrieve dimensions from shapes of xt and Wy
            n_x, m = xt.shape
            n_y, n_a = Wy.shape

            ### START CODE HERE ###
            # Concatenate a_prev and xt (≈1 line)
            concat = np.concatenate((a_prev, xt), axis=0)

            # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
            ft = sigmoid(Wf @ concat + bf)
            it = sigmoid(Wi @ concat + bi)
            cct = np.tanh(Wc @ concat + bc)
            c_next = ft * c_prev + it * cct
            ot = sigmoid(Wo @ concat + bo)
            a_next = ot * np.tanh(c_next)

            # Compute prediction of the LSTM cell (≈1 line)
            yt_pred = softmax(Wy @ a_next + by)
            ### END CODE HERE ###

            # store values needed for backward propagation in cache
            cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

            return a_next, c_next, yt_pred, cache
        ```

- Forward Pass for LSTM
    - iterate over the LSTM cell using a for loop to process a sequence of Tx inputs
    - Exercise 4 - lstm_forward
        - get the dimensions nx, na, ny, m, Tx from the shape of the variables x and params
        - initialize the 3D tensors a, c, and y
            - a: hidden state, shape (na, m, Tx)
            - c: cell state, shape (na, m, Tx)
            - y: prediction, shape (ny, m, Tx)
            - setting one variable equal to the other is a copy by reference
            - in other words, don’t do c=a, otherwise both variables point to the same underlying variable
        - initialize the 2D tensor a<t>
            - a<t> stores the hidden state for time step t
            - a<0>, the initial hidden state at time step 0, is passed in when calling the function
            - a<t> and a<0> represent a single time step, so they both have the shape (na, m)
            - initialize a<t> by setting it to the initial hidden state (a<0>) that is passed into the function
        - initialize c<t> with zeros
            - c<t> represents a single time step, so its shape is (na, m)
            - create c_next as its own variable with its own location in memory
            - do not initialize it as a slice of the 3D tensor c
        - for each time step
            - from the 3D tensor x, get a 2D slice x<t> at time step t
            - call the lstm_cell_forward function to get the hidden state, cell state, prediction, and cache
            - store the hidden state, cell state and prediction inside the 3D tensors
            - append the cache to the list of caches

        ```python
        # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: lstm_forward

        def lstm_forward(x, a0, parameters):
            """
            Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

            Arguments:
            x -- Input data for every time-step, of shape (n_x, m, T_x).
            a0 -- Initial hidden state, of shape (n_a, m)
            parameters -- python dictionary containing:
                                Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                                bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                                Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                                bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                                Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                                bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                                Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                                bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                                Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

            Returns:
            a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
            y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
            c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
            caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
            """

            # Initialize "caches", which will track the list of all the caches
            caches = []

            ### START CODE HERE ###
            Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
            # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
            n_x, m, T_x = x.shape
            n_y, n_a = Wy.shape

            # initialize "a", "c" and "y" with zeros (≈3 lines)
            a = np.zeros((n_a, m, T_x))
            c = np.zeros((n_a, m, T_x))
            y = np.zeros((n_y, m, T_x))

            # Initialize a_next and c_next (≈2 lines)
            a_next = a0
            c_next = np.zeros((n_a, m))

            # loop over all time-steps
            for t in range(T_x):
                # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
                xt = x[:,:,t]
                # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
                a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
                # Save the value of the new "next" hidden state in a (≈1 line)
                a[:,:,t] = a_next
                # Save the value of the next cell state (≈1 line)
                c[:,:,t] = c_next
                # Save the value of the prediction in y (≈1 line)
                y[:,:,t] = yt
                # Append the cache into caches (≈1 line)
                caches.append(cache)

            ### END CODE HERE ###

            # store values needed for backward propagation in cache
            caches = (caches, x)

            return a, y, c, caches
        ```

Backpropagation in Recurrent NN

- loss J backward to a is assumed to be calculated elsewhere and passed to rnn_backward
- further assumed that the loss has been adjusted for batch size m and division by the number of examples is not required
- Basic RNN Backward Pass
    - compute the backward pass for the basic RNN cell

    - shorthand for dJ/dWax is dWax

    - the da part is computed separately
    - to compute rnn_cell_backward, use the following equations

    - Exercise 5 - rnn_cell_backward
        - simplify the equations by computing dz and using the chain rule
        - tanh(Wax…) was computed and saved as a_next in the forward pass
        - to calculate dba, the batch above is a sum across all m examples (axis=1), use keepdims = True
        - matrix vector derivatives described [here](http://cs231n.stanford.edu/vecDerivs.pdf)

        ```python
        # UNGRADED FUNCTION: rnn_cell_backward

        def rnn_cell_backward(da_next, cache):
            """
            Implements the backward pass for the RNN-cell (single time-step).

            Arguments:
            da_next -- Gradient of loss with respect to next hidden state
            cache -- python dictionary containing useful values (output of rnn_cell_forward())

            Returns:
            gradients -- python dictionary containing:
                                dx -- Gradients of input data, of shape (n_x, m)
                                da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                                dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                                dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                                dba -- Gradients of bias vector, of shape (n_a, 1)
            """

            # Retrieve values from cache
            (a_next, a_prev, xt, parameters) = cache

            # Retrieve values from parameters
            Wax = parameters["Wax"]
            Waa = parameters["Waa"]
            Wya = parameters["Wya"]
            ba = parameters["ba"]
            by = parameters["by"]

            ### START CODE HERE ###
            # compute the gradient of dtanh term using a_next and da_next (≈1 line)
            dtanh = da_next * (1 - a_next ** 2)

            # compute the gradient of the loss with respect to Wax (≈2 lines)
            dxt = Wax.T @ dtanh
            dWax = dtanh @ xt.T

            # compute the gradient with respect to Waa (≈2 lines)
            da_prev = Waa.T @ dtanh
            dWaa = dtanh @ a_prev.T

            # compute the gradient with respect to b (≈1 line)
            dba = np.sum(dtanh, axis=1, keepdims=True)

            ### END CODE HERE ###

            # Store the gradients in a python dictionary
            gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

            return gradients
        ```

    - Exercise 6 - rnn_backward
        - computing the gradients of the cost w.r.t a<t> at every time step t is useful because it is what helps the gradient backprop to the previous RNN cell
        - iterate through all the time steps starting at the end, and at each step, increment the overall dba, dWaa, dWax and store dx
        - initialize the return variables with zeros first, then loop through all the time steps while calling the rnn_cell_backward at each time step, updating the other variables accordingly
        - the notebook does not implement the backward path from the Loss J backwards to a
            - this would have included the dense layer and softmax which are part of the forward path
            - this is assumed to be calculated elsewhere and the result passed to rnn_backward in da
            - combine this with the loss from the previous stages when calling rnn_cell_backward

        ```python
        # UNGRADED FUNCTION: rnn_backward

        def rnn_backward(da, caches):
            """
            Implement the backward pass for a RNN over an entire sequence of input data.

            Arguments:
            da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
            caches -- tuple containing information from the forward pass (rnn_forward)

            Returns:
            gradients -- python dictionary containing:
                                dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                                da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                                dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                                dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                                dba -- Gradient w.r.t the bias, of shape (n_a, 1)
            """

            ### START CODE HERE ###

            # Retrieve values from the first cache (t=1) of caches (≈2 lines)
            (caches, x) = caches
            (a1, a0, x1, parameters) = caches[0]

            # Retrieve dimensions from da's and x1's shapes (≈2 lines)
            n_a, m, T_x = da.shape
            n_x, m = x1.shape 

            # initialize the gradients with the right sizes (≈6 lines)
            dx = np.zeros((n_x, m, T_x))
            dWax = np.zeros((n_a, n_x))
            dWaa = np.zeros((n_a, n_a))
            dba = np.zeros((n_a, 1))
            da0 = np.zeros((n_a, m))
            da_prevt = np.zeros((n_a, m, T_x))

            # Loop through all the time steps
            for t in reversed(range(T_x)):
                # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. (≈1 line)
                gradients = rnn_cell_backward(da[:,:,t], caches[t])
                # Retrieve derivatives from gradients (≈ 1 line)
                dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
                # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
                dx[:, :, t] = dxt
                dWax += dWaxt
                dWaa += dWaat 
                dba += dbat

            # Set da0 to the gradient of a which has been backpropagated through all time-steps (≈1 line) 
            da0 = da_prevt
            ### END CODE HERE ###

            # Store the gradients in a python dictionary
            gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}

            return gradients
        ```

- LSTM Backward Pass
    - One Step Backward
        - slightly more complex

    - Gate Derivatives
        - note the location of the gate derivatives between the dense layer and activation function
        - this is convenient for computing parameter derivatives in the next step

    - Parameter Derivatives

        - to calc dbf, dbu, dbc, dbo, sum across all m examples (axis=1) on dy<t>f, dy<t>u, fp~c<t>, dy<t>o

        - compute the derivative w.r.t the previous hidden sate, previous memory state, and input

        - to account for concatenation, the weights for equations 19 are the first n_a

        - the weights are from n_a to the end
    - Exercise 7 - lstm_cell_backward

        ```python
        # UNGRADED FUNCTION: lstm_cell_backward

        def lstm_cell_backward(da_next, dc_next, cache):
            """
            Implement the backward pass for the LSTM-cell (single time-step).

            Arguments:
            da_next -- Gradients of next hidden state, of shape (n_a, m)
            dc_next -- Gradients of next cell state, of shape (n_a, m)
            cache -- cache storing information from the forward pass

            Returns:
            gradients -- python dictionary containing:
                                dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                                da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                                dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                                dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                                dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                                dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                                dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                                dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                                dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                                dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                                dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
            """

            # Retrieve information from "cache"
            (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

            ### START CODE HERE ###
            # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
            n_x, m = xt.shape
            n_a, m = a_next.shape

            # Compute gates related derivatives. Their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
            dot = da_next * np.tanh(c_next) * ot * (1-ot)
            dcct = (dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next) * (1 - cct ** 2)
            dit = (dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next) * it * (1 - it)
            dft = (dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev *da_next) * ft * (1 - ft)

            # Compute parameters related derivatives. Use equations (11)-(18) (≈8 lines)
            dWf = dft @ np.concatenate((a_prev, xt), axis=0).T
            dWi = dit @ np.concatenate((a_prev, xt), axis=0).T
            dWc = dcct @ np.concatenate((a_prev, xt), axis=0).T
            dWo = dot @ np.concatenate((a_prev, xt), axis=0).T
            dbf = np.sum(dft, axis=1, keepdims=True)
            dbi = np.sum(dit, axis=1, keepdims=True)
            dbc = np.sum(dcct, axis=1, keepdims=True)
            dbo = np.sum(dot, axis=1, keepdims=True)

            # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (19)-(21). (≈3 lines)
            da_prev = parameters['Wf'][:,:n_a].T @ dft + parameters['Wi'][:,:n_a].T @ dit + parameters['Wc'][:,:n_a].T @ dcct + parameters['Wo'][:,:n_a].T @ dot
            dc_prev = dc_next * ft + ot * (1 - np.tanh(c_next) ** 2) * ft * da_next
            dxt = parameters['Wf'][:,n_a:].T @ dft + parameters['Wi'][:,n_a:].T @ dit + parameters['Wc'][:,n_a:].T @ dcct + parameters['Wo'][:,n_a:].T @ dot
            ### END CODE HERE ###

            # Save gradients in dictionary
            gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                        "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

            return gradients
        ```

- Backward Pas through the LSTM RNN
    - first create variables of the same dimension as the return variables
    - then iterate over all the time steps starting form the end and call the step function
    - update the params by summing them individually
    - return a dictionary with the new grads
    - Exercise 8 - lstm_backward
        - create a for loop starting from Tx and going backward
        - for each step, call lstm_cell_backward and update the old gradients by adding the new gradients to them

        ```python
        # UNGRADED FUNCTION: lstm_backward

        def lstm_backward(da, caches):

            """
            Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

            Arguments:
            da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
            caches -- cache storing information from the forward pass (lstm_forward)

            Returns:
            gradients -- python dictionary containing:
                                dx -- Gradient of inputs, of shape (n_x, m, T_x)
                                da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                                dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                                dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                                dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                                dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                                dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                                dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                                dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                                dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
            """

            # Retrieve values from the first cache (t=1) of caches.
            (caches, x) = caches
            (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

            ### START CODE HERE ###
            # Retrieve dimensions from da's and x1's shapes (≈2 lines)
            n_a, m, T_x = da.shape
            n_x, m = x1.shape

            # initialize the gradients with the right sizes (≈12 lines)
            dx = np.zeros((n_x, m, T_x))
            da0 = np.zeros((n_a, m))
            da_prevt = np.zeros((n_a, m))
            dc_prevt = np.zeros((n_a, m))
            dWf = np.zeros((n_a, n_a+n_x))
            dWi = np.zeros((n_a, n_a+n_x))
            dWc = np.zeros((n_a, n_a+n_x))
            dWo = np.zeros((n_a, n_a+n_x))
            dbf = np.zeros((n_a, 1))
            dbi = np.zeros((n_a, 1))
            dbc = np.zeros((n_a, 1))
            dbo = np.zeros((n_a, 1))

            # loop back over the whole sequence
            for t in reversed(range(T_x)):
                # Compute all gradients using lstm_cell_backward. Choose wisely the "da_next" (same as done for Ex 6).
                gradients = lstm_cell_backward(da[:,:,t], dc_prevt, caches[t])
                # Store or add the gradient to the parameters' previous step's gradient
                da_prevt = gradients['da_prev']
                dc_prevt = gradients['dc_prev']
                dx[:,:,t] = gradients['dxt']
                dWf += gradients['dWf']
                dWi += gradients['dWi']
                dWc += gradients['dWc']
                dWo += gradients['dWo']
                dbf += gradients['dbf']
                dbi += gradients['dbi']
                dbc += gradients['dbc']
                dbo += gradients['dbo']
            # Set the first activation's gradient to the backpropagated gradient da_prev.
            da0 = gradients['da_prev']

            ### END CODE HERE ###

            # Store the gradients in a python dictionary
            gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                        "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

            return gradients
        ```