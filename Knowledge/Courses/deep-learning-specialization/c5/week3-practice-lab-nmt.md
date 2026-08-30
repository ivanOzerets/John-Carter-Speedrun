Packages

```python
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
%matplotlib inline
```

Translating Human Readable Dates Into Machine Readable Dates

- the model can be used to translate from one language to another
- the language translation requires massive datasets and usually takes days of training
- a simpler date translation task will be performed
- the network will input a date written in a variety of possible formats
- the network will translate then into standardized, machine readable dates
- the network will learn to output dates in the common machine-readable format YYYY-MM-DD
- Dataset
    - train the model on a dataset of 10k human readable dates and their equivalent, standardized, machine readable dates

        ```python
        m = 10000
        dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
        ```

        ```python
        dataset[:10]

        '''
        [('9 may 1998', '1998-05-09'),
         ('10.11.19', '2019-11-10'),
         ('9/10/70', '1970-09-10'),
         ('friday october 24 2025', '2025-10-24'),
         ('thursday february 26 2026', '2026-02-26'),
         ('monday august 19 2024', '2024-08-19'),
         ('saturday april 28 1990', '1990-04-28'),
         ('26 jan 1995', '1995-01-26'),
         ('7 mar 1983', '1983-03-07'),
         ('sunday may 22 1988', '1988-05-22')]
        '''
        ```

        - dataset: a list of tuples of (human readable date, machine readable date)
        - human_vocab: a python dictionary mapping all chars used in the human readable dates to an integer-values index
        - machine_vocab: a python dictionary mapping all chars used in machine readable dates to an integer-valued index
            - the indices are not necessarily conssistent with human_vocab
        - inv_machine_vocab: the inverse dictionary of machine_vocab, mapping from indices back to chars
    - preprocess the data and map the raw text data into the index values
        - set Tx=30
            - assume Tx is the max length of the human readable date
            - for a longer input, truncate it
        - set Ty = 10
            - YYYY-MM-DD is 10 chars long

        ```python
        Tx = 30
        Ty = 10
        X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

        print("X.shape:", X.shape)
        print("Y.shape:", Y.shape)
        print("Xoh.shape:", Xoh.shape)
        print("Yoh.shape:", Yoh.shape)

        '''
        X.shape: (10000, 30)
        Y.shape: (10000, 10)
        Xoh.shape: (10000, 30, 37)
        Yoh.shape: (10000, 10, 11)
        '''
        ```

    - X: a processed version of the human readable dates in the training set
        - each char in X is replaced by an index mapped to the char using human_vocab
        - each date is padded to ensure a length of Tx using a special char (<pad>)
        - X.shape = (m, Tx) where m is the number of training examples in a batch
    - Y: a processed version of the machine readable dates in the training set
        - each char is replaced by the index it is mapped to in machine_vocab
        - Y.shape = (m, Ty)
    - Xoh: one-hot version of Y
        - each index of X is converted to the one-hot representation
        - Xoh.shpe = (m, Tx, len(human_vocab))
    - Yoh: one-hot version of Y
        - each index in Y is converted to the one-hot representation
        - Yoh.shape = (m, Ty, len(machine_vocab))
        - len(machine_vocab) = 11 since there are 10 numeric digits and the - symbol
    - some examples of preprocessed training examples

        ```python
        index = 0
        print("Source date:", dataset[index][0])
        print("Target date:", dataset[index][1])
        print()
        print("Source after preprocessing (indices):", X[index])
        print("Target after preprocessing (indices):", Y[index])
        print()
        print("Source after preprocessing (one-hot):", Xoh[index])
        print("Target after preprocessing (one-hot):", Yoh[index])

        '''
        Source date: 9 may 1998
        Target date: 1998-05-09

        Source after preprocessing (indices): [12  0 24 13 34  0  4 12 12 11 36 36 36 36 36 36 36 36 36 36 36 36 36 36
         36 36 36 36 36 36]
        Target after preprocessing (indices): [ 2 10 10  9  0  1  6  0  1 10]

        Source after preprocessing (one-hot): [[0. 0. 0. ... 0. 0. 0.]
         [1. 0. 0. ... 0. 0. 0.]
         [0. 0. 0. ... 0. 0. 0.]
         ...
         [0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 0. 0. 1.]]
        Target after preprocessing (one-hot): [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
        '''
        ```

Neural Machine Translation with Attention

- the attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step
- Attention Mechanism
    - the diagram on the left shows the attention model
    - the diagram on the right shows what one attention step does to calculate the attention variables a<t,t’>
    - the attention variables a<t,t’> are used to compute the context variable context<t> for each timestep in the output (t = 1, …, Ty)

    - Pre-attention and Post-attention LSTMs on both sides of the attention mechanism
        - there are two separate LSMTs in the model: pre-attention and post-attention LSTMs
        - pre-attention Bi-LSTM is the one at the bottom of the picture is a bi-directional LSTM and comes before the attention mechanism
            - the attention mechanism is shown in the middle of the left-hand diagram
            - the pre-attention bi-LSTM goes through Tx time steps
        - post attention LSTM:  at the top of the diagram comes after the attention mechanism
            - the post-attention LSTM goes through Ty time steps
        - the post-attention LSTM passes the hidden state s<t> and cell state c<t> from one time step to the next
    - An LSTM has both a hidden state and cell state
        - originally using only a basic RNN for the post-attention sequence model
            - means that the state captured by the RNN was outputting only the hidden state s<t>
        - here, use an LSTM instead of a basic RNN
            - so the LSTM has both the hidden state s<t> and the cell state c<t>
    - Each time step does not use predictions from the previous time step
        - unlike previous text generation examples, in this model, the post-attention LSTM at time t does not take the previous time step’s prediction y<t-1> as input
        - the post-attention LSTM at time ‘t’ only takes the hidden state s<t> and cell state c<t> as input
        - model is designed like this because unlike language generation (where adjacent characters are highly correlated) there isn’t as strong a dependency between the previous character and the next character in a YYY-MM-DD date
    - Concatenation of hidden states from the forward and backward pre-attention LSTMs
        - a→<t>: hidden state of the forward-direction, pre-attention LSTM
        - a←<t>: hidden state of the forward-direction, pre-attention LSTM
        - a<t> = [a→<t>, a←<t>]: the concatenation of the activations of both the forward-direction and backward-direction of the pre-attention Bi-LSTM
    - Computing ‘energies’ e<t, t’> as a function of s<t-1> and a<t’>
        - e is called the energies variable
        - s<t-1> is the hidden state of the post-attention LSTM
        - a<t’> is the hidden state of the pre-attention LSTM
        - s<t-1> and a<t> are fed into a simple NN, which learn the function to output e<t,t’>
        - e<t,t’> is then used when computing the attention a<t,t’> that y<t> should pay to a<t’>
        - the diagram on the right of figure 1 uses a RepeatVector node to copy s<t-1>s value Tx times
        - then it uses concatenation to concatenate s<t-1> and a<t>
        - the concatenation of s<t-1> and a<t> is fed into a dense layer, which computes e<t,t’>
        - e<t,t’> is then passed through a softmax to compute a<t,t’>
        - the diagram doesn’t explicitly show variable e<t,t’>, but e<t,t’> is above dense layer and below the softmax layer in the diagram in the right half
    - start by implementing two function: one_step_attention() and model()
    - one_step_attention
        - the inputs to the one_step_attention at time step t are
            - [a<1>, a<2>, …, a<Tx>]: all hidden states of the pre-attention Bi-LSTM
            - s<t-1>: the previous hidden state of the post-attention LSTM
        - one_step_attention computes:
            - [alpha<t,1>, alpha<t,2>, …, alpha<t,Tx>]: the attention weights
            - context<t>: the context vector

    - clarifying context and c
        - context was denoted as c<t>
        - here, context is context<t>
        - this is to avoid confusion with the LSTM internal memory cell also denoted by c<t>
    - Exercise 1 - one_step_attention
        - the function model() will call the layers in one_step_attention() Ty times using a for loop
        - it is important that all Ty copies have the same weights
            - it should not reinitialize the weights every time
            - Ty steps should have shared weights
        - how to implement layers with shareable weights in Keras:
            - define the layer object in a variable scape that is outside of the one_step_attention function, for example, defining the objects as global variables would work
                - defining these variables inside the scope of the function model would technically work, since model will then call the one_step_attention function
                - here, they are global variables
            - call the objects when propagating the input
        - layers are defined as global variables
        - check documentation
            - RepeatVector(): var_repreated = repeat_layer(var1)
            - Concatenate(): concatenated_vars = concatenate_layer([var1, var2, var3])
            - Dense(): var_out = dense_layer(var_in)
            - Activation(): activation = activation_layer(var_in)
            - Dot(): dot_product = dot_layer([var1, var2])

        ```python
        # Defined shared layers as global variables
        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(10, activation = "tanh")
        densor2 = Dense(1, activation = "relu")
        activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
        dotor = Dot(axes = 1)
        ```

        ```python
        # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: one_step_attention

        def one_step_attention(a, s_prev):
            """
            Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
            "alphas" and the hidden states "a" of the Bi-LSTM.

            Arguments:
            a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
            s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

            Returns:
            context -- context vector, input of the next (post-attention) LSTM cell
            """

            ### START CODE HERE ###
            # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
            s_prev = repeator(s_prev)
            # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
            # For grading purposes, please list 'a' first and 's_prev' second, in this order.
            concat = concatenator([a, s_prev])
            # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
            e = densor1(concat)
            # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
            energies = densor2(e)
            # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
            alphas = activator(energies)
            # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
            context = dotor([alphas, a])
            ### END CODE HERE ###

            return context
        ```

    - Exercise 2 - modelf
        - modelf first runs the input through a Bi-LSTM to get [a<1>, a<2>, …, a<Tx>]
        - then, modelf calls one_step_attention() Ty times using a for loop
        - at each iteration of this loop
            - give the computed context vector context<t> to the post-attention LSTM
            - run the output of the post-attention LSTM through a dense layer with softmax activation
            - the softmax generates a prediction y_hat<t>

        ```python
        ### You cannot edit this cell

        n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
        n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

        # Please note, this is the post attention LSTM cell.  
        post_activation_LSTM_cell = LSTM(n_s, return_state = True) # Please do not modify this global variable.
        output_layer = Dense(len(machine_vocab), activation=softmax)
        ```

        - use these layers Ty times in a for loop to generate the outputs, and their params will not be reinitialized
        - propagate the input X into a bi-directional LSTM
            - [Bidirectional](https://keras.io/layers/wrappers/#bidirectional)
            - [LSTM](https://keras.io/layers/recurrent/#lstm)
            - want the LSTM to return a full sequence instead of just the last hidden state

            ```python
            sequence_of_hidden_states = Bidirectional(LSTM(units=..., return_sequences=...))(the_input_X)
            ```

        - iterate for t = 0, …, Ty - 1
            - call one_step_attention(), passing in the sequence of hidden states [a<1>, a<2>, …, a<Tx>] from the pre-attention bi-directional LSTM, and the previous hidden state s<t-1> from the post-attention LSTM to calc the context vector context<t>
            - give context<t> to the post-attention LSTM cell
                - remember to pass the previous hidden state s<t01> and cell-states c<t-1> of the LSTM
                - this outputs the new hidden state s<t> and the new cell state c<t>

                ```python
                _, next_hidden_state, next_cell_state = 
                  post_activation_LSTM_cell(inputs=..., initial_state=[prev_hidden_state, prev_cell_state])
                ```

                - the layer is actually the post attention LSTM cell
            - apply a dense, softmax layer to s<t>, get the output

                ```python
                output = output_layer(inputs=...)
                ```

            - save the output by adding it to the list of outputs
        - Create the Keras model instance
            - should have three inputs
                - X, the one-hot encoded inputs to the model, of shape (Tx, humanVocabSize)
                - s<0>, the initial hidden state of the post-attention LSTM
                - c<0>, the initial cell state of the post-attention LSTM
            - the output is the list of outputs

                ```python
                model = Model(inputs=[...,...,...], outputs=...)
                ```

        ```python
        # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: model

        def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size=None):
            """
            Arguments:
            Tx -- length of the input sequence
            Ty -- length of the output sequence
            n_a -- hidden state size of the Bi-LSTM
            n_s -- hidden state size of the post-attention LSTM
            human_vocab_size -- size of the python dictionary "human_vocab"
            machine_vocab_size -- integer, optional, size of the python dictionary "machine_vocab"
                                  This is not being used

            Returns:
            model -- Keras model instance
            """

            # Define the inputs of your model with a shape (Tx, human_vocab_size)
            # Define s0 (initial hidden state) and c0 (initial cell state)
            # for the decoder LSTM with shape (n_s,)
            X = Input(shape=(Tx, human_vocab_size))
            # initial hidden state
            s0 = Input(shape=(n_s,), name='s0')
            # initial cell state
            c0 = Input(shape=(n_s,), name='c0')
            # hidden state
            s = s0
            # cell state
            c = c0

            # Initialize empty list of outputs
            outputs = []

            ### START CODE HERE ###

            # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
            a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)

            # Step 2: Iterate for Ty steps
            for t in range(Ty):

                # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
                context = one_step_attention(a, s)

                # Step 2.B: Apply the post-attention LSTM cell to the "context" vector. (≈ 1 line)
                # Don't forget to pass: initial_state = [hidden state, cell state] 
                # Remember: s = hidden state, c = cell state
                _, s, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c])

                # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
                out = output_layer(inputs=s)

                # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
                outputs.append(out)

            # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
            model = Model(inputs=[X, s0, c0], outputs=outputs)

            ### END CODE HERE ###

            return model
        ```

        - unit test

        ```python
        ### You cannot edit this cell

        # UNIT TEST 2

        def modelf_states_test(target):
            Tx = 30
            n_a = 32
            n_s = 64
            len_human_vocab = 37

            model = target(Tx, Ty, n_a, n_s, len_human_vocab)

            # Create test inputs
            X_test = np.random.rand(1, Tx, len_human_vocab)
            s0_test = np.zeros((1, n_s))
            c0_test = np.zeros((1, n_s))

            @tf.function(input_signature=[
                tf.TensorSpec(shape=[None, Tx, len_human_vocab], dtype=tf.float32),
                tf.TensorSpec(shape=[None, n_s], dtype=tf.float32),
                tf.TensorSpec(shape=[None, n_s], dtype=tf.float32)
            ])
            def predict_function(X, s0, c0):
                # Call the model directly with input tensors
                return model([X, s0, c0])  

            # Get the outputs of the model for the first five time steps
            outputs = predict_function(X_test, s0_test, c0_test)

            # Extract the hidden states (s) from the LSTM outputs for each time step
            hidden_states = [np.array(output) for output in outputs]

            # Check if consecutive hidden states are different
            for i in range(len(hidden_states) - 1):
                assert not np.allclose(hidden_states[i], hidden_states[i + 1]), (
                    "Consecutive hidden states should be different.\n"
                    "Check if the LSTM cell is using the correct previous states.\n"
                    "Make sure you are using s and c, and NOT using s0 and c0 in Step 2.B."
                )

            print("\033[32mAll tests passed!\033[0m")

        modelf_states_test(modelf)
        ```

        - create the model

            ```python
            ## You cannot edit this cell
            model = modelf(Tx, Ty, n_a, n_s, len(human_vocab))
            ```

        - summary

            ```python
            model.summary()

            '''
            Model: "functional_7"
            __________________________________________________________________________________________________
            Layer (type)                    Output Shape         Param #     Connected to                     
            ==================================================================================================
            input_12 (InputLayer)           [(None, 30, 37)]     0                                            
            __________________________________________________________________________________________________
            s0 (InputLayer)                 [(None, 64)]         0                                            
            __________________________________________________________________________________________________
            bidirectional_10 (Bidirectional (None, 30, 64)       17920       input_12[0][0]                   
            __________________________________________________________________________________________________
            repeat_vector_2 (RepeatVector)  (None, 30, 64)       0           s0[0][0]                         
            ...
            __________________________________________________________________________________________________
            dense_6 (Dense)                 (None, 11)           715         lstm_7[30][1]                    
                                                                             lstm_7[31][1]                    
                                                                             lstm_7[32][1]                    
                                                                             lstm_7[33][1]                    
                                                                             lstm_7[34][1]                    
                                                                             lstm_7[35][1]                    
                                                                             lstm_7[36][1]                    
                                                                             lstm_7[37][1]                    
                                                                             lstm_7[38][1]                    
                                                                             lstm_7[39][1]                    
            ==================================================================================================
            Total params: 52,960
            Trainable params: 52,960
            Non-trainable params: 0
            __________________________________________________________________________________________________

            '''
            ```

    - Exercise 3 - Compile the Model
        - after creating the model in Keras, compile it and define the loss function, optimizer and metrics
            - loss function: Adam
                - learning rate = 0.005
                - beta1 = 0.9
                - beta2 = 0.999
                - decay = 0.01
            - metric: ‘accuracy’

            ```python
            optimizer = Adam(lr=..., beta_1=..., beta_2=..., decay=...)
            model.compile(optimizer=..., loss=..., metrics=[...])
            ```

        ```python
        ### START CODE HERE ### (≈2 lines)
        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01) # Adam(...) 
        model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        ### END CODE HERE ###
        ```

    - Define inputs and outputs, and fit the model
        - input Xoh of shape (m = 10000, Tx = 30, human_vocab = 37) containing the training examples
        - create s0 and c0 to initialize the post_attention_LSTM_cell with zeros
        - given the model(), the outputs need to be a list of 10 elements of shape (m, T_y)
            - the list outputs[i][0], …, outputs[i][Ty} represents the true labels (chars) corresponding to the ith training examples (Xoh[i])
            - outputs[i][j] is the true label of the jth char in the ith training example

        ```python
        ### You cannot edit this cell
        s0 = np.zeros((m, n_s))
        c0 = np.zeros((m, n_s))
        outputs = list(Yoh.swapaxes(0,1))
        ```

        ```python
        ### You cannot edit this cell
        model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
        ```

        - tables gives an example of what the accuracies could be if the batch had two examples

            - thus, dense_2_8_accuracy: 0.89 means that predicting the ninth char of the outputs correctly 89% of the time in the current batch of data

        ```python
        ### You cannot edit this cell
        model.load_weights('models/model.h5')
        ```

        ```python
        ### You cannot edit this cell

        EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
        s00 = np.zeros((1, n_s))
        c00 = np.zeros((1, n_s))
        for example in EXAMPLES:
            source = string_to_int(example, Tx, human_vocab)
            #print(source)
            source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
            source = np.swapaxes(source, 0, 1)
            source = np.expand_dims(source, axis=0)
            prediction = model.predict([source, s00, c00])
            prediction = np.argmax(prediction, axis = -1)
            output = [inv_machine_vocab[int(i)] for i in prediction]
            print("source:", example)
            print("output:", ''.join(output),"\n")

        '''
        source: 3 May 1979
        output: 1979-05-33 

        source: 5 April 09
        output: 2009-04-05 

        source: 21th of August 2016
        output: 2016-08-20 

        source: Tue 10 Jul 2007
        output: 2007-07-10 

        source: Saturday May 9 2018
        output: 2018-05-09 

        source: March 3 2001
        output: 2001-03-03 

        source: March 3rd 2001
        output: 2001-03-03 

        source: 1 March 2001
        output: 2001-03-01 
        '''
        ```

Visualizing Attention

- since the problem has a fixed output length of ten, it is also possible to carry out this task using ten different softmax units to generate the ten chars of the output
- but one advantage of the attention model is that each part of the output (such as the month) knows it needs to depend only on a small part of the input (the chars in the input giving the month)
- possible to visualize what each part of the output is looking at which part of the input
- considering the task of translating ‘saturday 9 may 2018’ to ‘2018-05-09’
- can be visualized with the computed a<t,t’>

- the outputs the ‘saturday’ portion of the input
- none of the output timesteps are paying much attention to that portion of the input
- 9 has been translated as 09 and may has been correctly translated into 05, with the output paying attention to the parts of the input it needs to make the translation
- the year mostly requires it to pay attention to the input’s 18 in order to generate 2018
- Getting the Attention Weights From the Network
    - visualize the attention values in the network
    - propagate an example through the network, then visualize the values of a<t,t’>
    - figure out where the attention values are located, start by printing a summary of the model

        ```python
        model.summary()

        '''
        Model: "functional_7"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_12 (InputLayer)           [(None, 30, 37)]     0                                            
        __________________________________________________________________________________________________
        s0 (InputLayer)                 [(None, 64)]         0                                            
        __________________________________________________________________________________________________
        bidirectional_10 (Bidirectional (None, 30, 64)       17920       input_12[0][0]                   
        __________________________________________________________________________________________________
        repeat_vector_2 (RepeatVector)  (None, 30, 64)       0           s0[0][0]                         
        ...
        __________________________________________________________________________________________________
        dense_6 (Dense)                 (None, 11)           715         lstm_7[30][1]                    
                                                                         lstm_7[31][1]                    
                                                                         lstm_7[32][1]                    
                                                                         lstm_7[33][1]                    
                                                                         lstm_7[34][1]                    
                                                                         lstm_7[35][1]                    
                                                                         lstm_7[36][1]                    
                                                                         lstm_7[37][1]                    
                                                                         lstm_7[38][1]                    
                                                                         lstm_7[39][1]                    
        ==================================================================================================
        Total params: 52,960
        Trainable params: 52,960
        Non-trainable params: 0
        __________________________________________________________________________________________________

        '''
        ```

    - navigate through the output of model.summary()
    - see that the layer named attention_weights outputs the alphas of shape (m, 30, 1) before dot_2 computes the context vector for every times step t = 0, …, Ty - 1
    - the function plot_attention_map() pulls out the attention values from the model and plots them

        ```python
        attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);
        ```

    - on the generated plot, observe the values of the attention weights for each char of the predicted output
    - examine this plot and check that the places where the network is paying attention makes sense
    - most of the time attention helps predict the year, and doesn’t have much impact on predicting the day or month