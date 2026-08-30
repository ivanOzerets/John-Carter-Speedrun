Packages

```python
import IPython
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from music21 import *
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from outputs import *
from test_utils import *

from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
```

Problem Statement

- create a jazz music piece specially for a friend’s bday
- train a network to generate novel jazz solos in a style representative of a body of performed work
- Dataset
    - train the alg on a corpus of Jazz music

        ```python
        IPython.display.Audio('./data/30s_seq.wav')
        ```

    - the preprocessing of the musical data has been done beforehand
    - What are musical values
        - think of each value as a note, which comprises a pitch and duration
        - for example, pressing down on a specific piano key for 0.5 sec, then a note has been played
        - in music theory, a value is actually more complicated than this, specifically, it also captures the information needed to play multiple notes at the same time
        - for example, when playing a music piece, press down two piano keys at the same time (chord)
    - Music as a sequence of values
        - obtain a dataset of values, and use an RNN model to generate sequences of values
        - the music generation system will use 90 unique values
        - load raw music data and preprocess it into values

        ```python
        X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid')
        print('number of training examples:', X.shape[0])
        print('Tx (length of sequence):', X.shape[1])
        print('total # of unique values:', n_values)
        print('shape of X:', X.shape)
        print('Shape of Y:', Y.shape)
        print('Number of chords', len(chords))

        '''
        number of training examples: 60
        Tx (length of sequence): 30
        total # of unique values: 90
        shape of X: (60, 30, 90)
        Shape of Y: (30, 60, 90)
        Number of chords 19
        '''
        ```

    - just loaded the following
        - X: this is an (m, Tx, 90) dimensional array
            - you have m training example, each of which is a snippet of Tx = 30 musical values
            - at each time step, the input is one of the 90 different possible values, represented as a one-hot vector
                - for example, X[i,t,:] is a one-hot vector representing the value of the ith example at time t
        - Y: a (Ty, m, 90) dimensional array
            - this is essentially the same as X, but shifted one step to the left
            - notice that the data in Y is reordered to be dimension (Ty, m, 90), where Ty = Tx
            - this format makes it more convenient to feed into the LSTM later
            - use the previous values to predict the next value
                - sot he sequence model will try to predict y<t> given x<1>, …, x<t>
        - n_values: the number of unique values in this dataset, this should be 90
        - indices_values: python dictionary mapping ints 0 through 89 to musical values
        - chords: chords used in the input midi
- Model Overview
    - here is the architecture of the model

    - X = (x<1>, x<2>, …, x<Tx>) is a window of size Tx scanned over the musical corpus
    - each x<t> is an index corresponding to a value
    - y_hat<t> is the prediction for the next value
    - train the model on random snippets of 30 values taken from a much longer piece of music
        - thus, don’t bother to set the first input x<1> = 0, since most of these snippets of audio start somewhere in the middle of a piece of music
        - each snippets is set to have the same length Tx = 30 to make vectorization easier

Overview of Section 2 and 3

- train a model that predicts the next note in a style similar to the jazz music it’s trained on
- the training is contained in the weights and biases of the model
- then use the weights and biases in a new model that predicts a series of notes, and using the previous note to predict the next note
- the weights and biases are transferred to the new model using the global shared layers

Building the Model

- the model takes input X of shape(m, Tx, 90) and labels Y of shape (Ty, m, 90)
- use an LSTM with hidden states that have n_a = 64 dimensions

    ```python
    # number of dimensions for the hidden state of each LSTM cell.
    n_a = 64 
    ```

- Sequence generation uses a for-loop
    - when building an RNN where, at test time, the entire input sequence x<1>, x<2>, …, x<Tx> is given in advance, then Keras has simple build-in functions to build the model
    - however, for sequence generation, at test time, all x<t> values will not be known in advance
    - instead, generate them one at ta time using x<t> = y<t-1>
        - the input at time t is the prediction at the previous time step t-1
    - implement a for-loop to iterate over the time steps
- Shareable weights
    - the function djmodel() will call the LSTM layer Tx times using a for-loop
    - it is important that ll Tx copies have the same weights
    - referencing a globally defined shared layer will utilize the same layer-object instance at each time step
    - define the layer objects (global variables)
    - call the objects when propagating the input
- 3 types of layers
    - the layer objects needed for global variables have been defined
    - read Keras documentation and understand the layers:
        - reshape()
        - LSTM()
        - Dense()

    ```python
    n_values = 90 # number of music values
    reshaper = Reshape((1, n_values))                  # Used in Step 2.B of djmodel(), below
    LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
    densor = Dense(n_values, activation='softmax')     # Used in Step 2.D
    ```

    - reshaper, LSTM_cell and densor are globally defined layer objects that are used to implement djmodel()
    - in order to propagate a Keras tensor onject X through one of these layers, use layer_object()
        - for one input, use layer_object(x)
        - for more than one input, put the inputs in a list: layer_object([X1, X2)]
- Exercise 1 - djmodel
    - Inputs (given)
        - the Input() layer is used for defining the input X as well as the initial hidden state ‘a0’ and cell state c0
        - the shape param takes a tuple that does not include the batch dimension (m)

            ```python
            X = Input(shape=(Tx, n_values)) # X has 3 dimensions and not 2: (m, Tx, n_values)
            ```

    - Step 1: Outputs
        - create an empty list outputs to save the outputs of the LSTM Cell at every time step
    - Step 2: Loop through time steps
        - loop for t in 1, .., Tx
    - 2A: select the t time-set vector from X
        - X has the shape (m, Tx, n_values)
        - the shape of the t selection should be (n_values,)
        - if implementing in np instead of Keras, extract a slice from a 3D numpy array

            ```python
            var1 = array1[:,1,:]
            ```

    - 2B: reshape x to be (1, n_values)
        - use the reshaper() layer
        - this is a function that takes the previous layer as its input argument
    - 2C: run x through one step of LSTM_cell
        - initialize the LSTM_cell with the previous step’s hidden state a and cell state c
        - use the following formatting

            ```python
            _, next_hidden_state, next_cell_state = LSTM_cell(inputs=input_x, initial_state=[previous_hidden_state, previous_cell_state])
            ```

        - choose appropriate variables for inputs, hidden state and cell state
    - 2D: Dense layer
        - propagate the LSTM’s hidden state through a dense+softmax layer using densor
    - 2E: Append output
        - append the output to the list of outputs
    - Step 3: after the loop, create the model
        - use the keras model object to create a model
        - there are two ways to instantiate the Model object
        - one is by subclassing, which will not be used here
        - instead, use the highly flexible Functional API
        - with Function API, start from the input, then specify the model’s forward pass with chained layer calls, and finally create the model from inputs and outputs
        - specify the inputs and output like so

            ```python
            model = Model(inputs=[input_x, initial_hidden_state, initial_cell_state], outputs=the_outputs)
            ```

        - then, choose the appropriate variables for the input tensor, hidden state, cell state, and output

            ```python
            # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
            # GRADED FUNCTION: djmodel

            def djmodel(Tx, LSTM_cell, densor, reshaper):
                """
                Implement the djmodel composed of Tx LSTM cells where each cell is responsible
                for learning the following note based on the previous note and context.
                Each cell has the following schema: 
                        [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
                Arguments:
                    Tx -- length of the sequences in the corpus
                    LSTM_cell -- LSTM layer instance
                    densor -- Dense layer instance
                    reshaper -- Reshape layer instance

                Returns:
                    model -- a keras instance model with inputs [X, a0, c0]
                """
                # Get the shape of input values
                n_values = densor.units

                # Get the number of the hidden state vector
                n_a = LSTM_cell.units

                # Define the input layer and specify the shape
                X = Input(shape=(Tx, n_values)) 

                # Define the initial hidden state a0 and initial cell state c0
                # using `Input`
                a0 = Input(shape=(n_a,), name='a0')
                c0 = Input(shape=(n_a,), name='c0')
                a = a0
                c = c0
                ### START CODE HERE ### 
                # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
                outputs = []

                # Step 2: Loop over tx
                for t in range(Tx):

                    # Step 2.A: select the "t"th time step vector from X. 
                    x = X[:,t,:]
                    # Step 2.B: Use reshaper to reshape x to be (1, n_values) (≈1 line)
                    x = reshaper(x)
                    # Step 2.C: Perform one step of the LSTM_cell
                    _, a, c = LSTM_cell(inputs=x, initial_state=[a,c])
                    # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
                    out = densor(a)
                    # Step 2.E: append the output to "outputs"
                    outputs.append(out)

                # Step 3: Create model instance
                model = Model(inputs=[X, a0, c0], outputs=outputs)

                ### END CODE HERE ###

                return model
            ```

    - Create the model object
        - define the model
        - use Tx = 30

        ```python
        ### YOU CANNOT EDIT THIS CELL

        model = djmodel(Tx=30, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)
        ```

        ```python
        # Check your model
        model.summary()

        '''
        Model: "functional_7"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_5 (InputLayer)            [(None, 30, 90)]     0                                            
        __________________________________________________________________________________________________
        tf_op_layer_strided_slice_91 (T [(None, 90)]         0           input_5[0][0]                    
        __________________________________________________________________________________________________
        reshape_1 (Reshape)             (None, 1, 90)        0           tf_op_layer_strided_slice_91[0][0
                                                                         tf_op_layer_strided_slice_92[0][0
        ...                    
                                                                         lstm_1[27][1]                    
                                                                         lstm_1[28][1]                    
                                                                         lstm_1[29][1]                    
        ==================================================================================================
        Total params: 45,530
        Trainable params: 45,530
        Non-trainable params: 0
        __________________________________________________________________________________________________

        '''
        ```

    - compile the model for training
        - compile the model to be trained
        - use the adam optimizer
        - loss function: categorical cross-entropy

        ```python
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        ```

    - Initialize hidden state and cell state
        - initialize a0 and c0 for the LSTM’s initial state to be zero

        ```python
        m = 60
        a0 = np.zeros((m, n_a))
        c0 = np.zeros((m, n_a))
        ```

    - Train the model
        - turn T into a list, since the cost function expects Y to be provided in this format
        - list(Y) is a list with 30 items, where each of the list items is of shape (60, 90)
        - train for 100 epochs

        ```python
        history = model.fit([X, a0, c0], list(Y), epochs=100, verbose = 0)
        ```

        ```python
        print(f"loss at epoch 1: {history.history['loss'][0]}")
        print(f"loss at epoch 100: {history.history['loss'][99]}")
        plt.plot(history.history['loss'])

        '''
        loss at epoch 1: 129.85726928710938
        loss at epoch 100: 8.811135292053223
        '''
        ```

Generating Music

- Predicting and Sampling

    - at each step of sampling
        - take as input the activation ‘a’ and cell state ‘c’ from the previous state of the LSTM
        - forward prop by one step
        - get a new output activation, as well as cell state
        - the new activation ‘a’ can then be used to generate the output using the fully connected layer, densor
    - Initialization
        - initialize the following to be zeros
            - x0
            - hidden state a0
            - cell state c0
    - Exercise 2 - music_inference_model
        - sample a sequence of musical values
        - Step 1: create an empty list outputs to save the outputs of the LSTM cell at every time step
        - Step 2A: use LSTM_Cell, which takes in the input layer, as well as the previous step’s ‘c’ and ‘a’ to generate the current step’s ‘c’ and ‘a’

            ```python
            _, next_hidden_state, next_cell_state = LSTM_cell(input_x, initial_state=[previous_hidden_state, previous_cell_state])
            ```

        - Step 2B: compute the output by applying densor to compute a softmax on ‘a’ to get the output for the current step
        - Step 2C: append the output to the list outputs
        - Step 2D: convert the last output into a new input for the next time step
            - get the index of the maximum value of the predicted output using tf.math.argmax along the last axis
            - convert the index into its n_values-one-hot encoding using tf.one_hot
        - Step 2E: use RepeatVector(1)(x) to convert the output of the one-hot encoding with shape=(None, 90) into a tensor with shape=(None, 1, 90)
        - Step 3: inference Model:
            - use the Keras Model object

            ```python
            model = Model(inputs=[input_x, initial_hidden_state, initial_cell_state], outputs=the_outputs)
            ```

        ```python
        # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: music_inference_model

        def music_inference_model(LSTM_cell, densor, Ty=100):
            """
            Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

            Arguments:
            LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
            densor -- the trained "densor" from model(), Keras layer object
            Ty -- integer, number of time steps to generate

            Returns:
            inference_model -- Keras model instance
            """

            # Get the shape of input values
            n_values = densor.units
            # Get the number of the hidden state vector
            n_a = LSTM_cell.units

            # Define the input of your model with a shape 
            x0 = Input(shape=(1, n_values))

            # Define s0, initial hidden state for the decoder LSTM
            a0 = Input(shape=(n_a,), name='a0')
            c0 = Input(shape=(n_a,), name='c0')
            a = a0
            c = c0
            x = x0

            ### START CODE HERE ###
            # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
            outputs = []

            # Step 2: Loop over Ty and generate a value at every time step
            for t in range(Ty):
                # Step 2.A: Perform one step of LSTM_cell. Use "x", not "x0" (≈1 line)
                _, a, c = LSTM_cell(x, initial_state=[a, c])

                # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                out = densor(a)
                # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 90) (≈1 line)
                outputs.append(out)

                # Step 2.D: 
                # Select the next value according to "out",
                # Set "x" to be the one-hot representation of the selected value
                # See instructions above.
                x = tf.math.argmax(out, axis=1)
                x = tf.one_hot(x, n_values)
                # Step 2.E: 
                # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
                x = RepeatVector(1)(x)

            # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
            inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

            ### END CODE HERE ###

            return inference_model
        ```

        ```python
        ### YOU CANNOT EDIT THIS CELL
        inference_model = music_inference_model(LSTM_cell, densor, Ty = 50)
        ```

        ```python
        # Check the inference model
        inference_model.summary()

        '''
        Model: "functional_21"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_15 (InputLayer)           [(None, 1, 90)]      0                                            
        __________________________________________________________________________________________________
        a0 (InputLayer)                 [(None, 64)]         0                                            
        __________________________________________________________________________________________________
        c0 (InputLayer)                 [(None, 64)]         0                                            
        __________________________________________________________________________________________________
        lstm_3 (LSTM)                   [(None, 64), (None,  39680       input_15[0][0]                   
        ...
        tf_op_layer_ArgMax_300 (TensorF [(None,)]            0           dense_3[78][0]                   
        __________________________________________________________________________________________________
        tf_op_layer_OneHot_298 (TensorF [(None, 90)]         0           tf_op_layer_ArgMax_300[0][0]     
        __________________________________________________________________________________________________
        repeat_vector_298 (RepeatVector (None, 1, 90)        0           tf_op_layer_OneHot_298[0][0]     
        ==================================================================================================
        Total params: 45,530
        Trainable params: 45,530
        Non-trainable params: 0
        __________________________________________________________________________________________________
        '''
        ```

    - Initialize inference model
        - create the zero-valued vectors used to initialize x and the LSTM state variables a and c

            ```python
            x_initializer = np.zeros((1, 1, n_values))
            a_initializer = np.zeros((1, n_a))
            c_initializer = np.zeros((1, n_a))
            ```

    - Exercise 3 - predict_and_sample
        - this function takes many args, including the inputs x_initializer, a_initializer, and c_initializer
        - in order to predict the output corresponding to this input, carry out 3 steps
        - Step 1
            - use infrerence model to predict an output given the set of inputs
            - the output pred should be a list of length Ty where each element is a numpy-array of shape (1, n_values)

            ```python
            inference_model.predict([input_x_init, hidden_state_init, cell_state_init])
            ```

            - Choose the appropriate input args to predict from the input args of this function
        - Step 2
            - convert pred into numpy array of Ty indices
                - each index is computed by taking the argmax of an element of the pred list
                - use np.argmax
                - set the axis parameter
                    - the shape of the prediction is (m, Ty, n_values)
        - Step 3
            - convert the indices into their one-hot vector representations
            - use to_categorical
            - set the num_classes parameter
            - use a dimension from the given parameters of predict_and_sample() (for example, one of the dimensions of x_initializer has the value for the number of distinct classes

        ```python
        # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: predict_and_sample

        def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                               c_initializer = c_initializer):
            """
            Predicts the next value of values using the inference model.

            Arguments:
            inference_model -- Keras model instance for inference time
            x_initializer -- numpy array of shape (1, 1, 90), one-hot vector initializing the values generation
            a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
            c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

            Returns:
            results -- numpy-array of shape (Ty, 90), matrix of one-hot vectors representing the values generated
            indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
            """

            n_values = x_initializer.shape[2]

            ### START CODE HERE ###
            # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
            pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
            # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
            indices = np.argmax(pred, axis=2)
            # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
            results = to_categorical(indices, num_classes=n_values)
            ### END CODE HERE ###

            return results, indices
        ```

        ```python
        ### YOU CANNOT EDIT THIS CELL

        results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

        print("np.argmax(results[12]) =", np.argmax(results[12]))
        print("np.argmax(results[17]) =", np.argmax(results[17]))
        print("list(indices[12:18]) =", list(indices[12:18]))

        '''
        np.argmax(results[12]) = 74
        np.argmax(results[17]) = 19
        list(indices[12:18]) = [array([74]), array([66]), array([68]), array([17]), array([87]), array([19])]
        '''
        ```

- Generate Music
    - the RNN generates a sequence of values
    - first call the predict_and_sample() function
    - the values are then post-processed into musical chords
    - most computational music algs use some post-processing because it’s difficult to generate music that sounds good without it
    - the post-processing does things like clean up the generated audio by making sure the same sound is not repeated too many times, or that two successive notes are not too far from each other in pitch, …
    - once could argue that a lot of these post-processing steps arehacks; also, a lot of the music generation literature has also fucused on hand-crafting post-processors, and a lot of the output quality depends on the quality of the post-processing and not just the quality of the model

    ```python
    out_stream = generate_music(inference_model, indices_values, chords)

    '''
    Predicting new values for different set of chords.
    Generated 32 sounds using the predicted values for the set of chords ("1") and after pruning
    Generated 32 sounds using the predicted values for the set of chords ("2") and after pruning
    Generated 32 sounds using the predicted values for the set of chords ("3") and after pruning
    Generated 32 sounds using the predicted values for the set of chords ("4") and after pruning
    Generated 32 sounds using the predicted values for the set of chords ("5") and after pruning
    Your generated music is saved in output/my_music.midi
    '''
    ```

    - using basic midi to wav parser, generate a rough idea about the audio clip generated by the model, the parser is very limited

    ```python
    mid2wav('output/my_music.midi')
    IPython.display.Audio('./output/rendered.wav')
    ```