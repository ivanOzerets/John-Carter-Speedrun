- task: leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth
- job is to give names to these dinosaurs
- to create new dinosaur names, build a character-level language model to generate new names
- learn the different name patterns, and randomly generate new names

Packages

```python
import numpy as np
from utils import *
import random
import pprint
import copy
```

Problem Statement

- Dataset and Preprocessing
    - read the dataset of dinosaur names, create a list of unique characters, and compute the dataset and vocab size

        ```python
        data = open('dinos.txt', 'r').read()
        data= data.lower()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

        # There are 19909 total characters and 27 unique characters in your data.

        ```

    - the characters are a-z plus the \n (newline char)
    - the newline char plays a role similar to the <EOS> token
    - char_to_ix: create a python dictionary to map each character to an index from 0-26
    - ix_to_char: create a second Python dictionary that maps each index back to the corresponding char
        - helps figure out which index corresponds to which char in the prob distribution output of the softmax layer

        ```python
        chars = sorted(chars)
        print(chars)

        # ['\n', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        ```

        ```python
        char_to_ix = { ch:i for i,ch in enumerate(chars) }
        ix_to_char = { i:ch for i,ch in enumerate(chars) }
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(ix_to_char)

        '''
        {   0: '\n',
            1: 'a',
            2: 'b',
            3: 'c',
            4: 'd',
        ...
            22: 'v',
            23: 'w',
            24: 'x',
            25: 'y',
            26: 'z'}

        '''
        ```

- Overview of the Model
    - initialize params
    - run the optimization loop
        - forward prop to compute the loss function
        - backward prop to compute the gradients w.r.t. the loss function
        - clip the grads to avoid exploding grads
        - using the grads, update the params with the grad descent update rule
    - return the learned params

    - at each time-step, the RNN tries to predict what the next char is, given the previous char
    - X = (x<1>, x<2>, …, x<Tx>) is a list of chars from the training set
    - Y = (y<1>, y<2>, …, y<Tx>) is the same list of chars but shifter one char forward
    - at every time-step t, y<t> = x<t+1>
    - the prediction at time t is the same as the input at time t + 1

Building Blocks of the Model

- gradient clipping: to avoid exploding gradients
- sampling: a technique used to generate chars
- Clipping the Gradients in the Optimization Loop
    - implement the clip function
    - exploding gradients
        - when gradients are very large
        - exploding grads make the training process more difficult, because the updates may be so large tat they overshoot the optimal values during back prop
    - loop structure
        - forward pass
        - cost computation
        - backward pass
        - param update
    - before updating the params, perform the grad clipping
    - function clip takes in a dictionary of grads and returns a clipped version of gradients
        - different ways to clip grads
        - use a simple element-wise clipping procedure, in which every element of the gradient vector is clipped to fall between some range [-N, N]
        - for example, if N=10
            - the range is [-10,10]
            - if any component of the grad vector is grater than 10, it is set to 19
            - less than -10, set to -10
            - in between, original values are kept

    - Exercise 1 - clip
        - return the clipped grads of the dictionary gradients
        - the function takes in a max threshold and returns the clipped versions of the grads
        - use np.clip
            - use params “out = …”
            - using the “out” param allows to update a variable “in-place”
            - not using “out”, the clipped var is stored in the var gradient but does not update the gradient variables

        ```python
        # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        ### GRADED FUNCTION: clip

        def clip(gradients, maxValue):
            '''
            Clips the gradients' values between minimum and maximum.

            Arguments:
            gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
            maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

            Returns: 
            gradients -- a dictionary with the clipped gradients.
            '''
            gradients = copy.deepcopy(gradients)

            dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

            ### START CODE HERE ###
            # Clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
            for gradient in gradients:
                np.clip(gradients[gradient], -maxValue, maxValue, out = gradients[gradient])
            ### END CODE HERE ###

            gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

            return gradients
        ```

- Sampling
    - assume that the model is trained, and would like to generate new text

    - Exercise 2 - sample
        - carry out four steps
        - step 1: input the dummy vector of zeros x<1> = 0
            - this is the default input before generating any chars
            - also set a<0> to 0
        - step 2: run one step of forward prop to get a<1> and y_hat<1>
        - hidden state

        - activation

        - prediction

        - note that y_hat<t+1> is a probability vector
        - y_hat<t+1>i represents the prob that the char indexed by i is the next char
        - step 3: sampling
            - with y<t+1>, select the next letter in the dinosaur name
            - selecting the most probable, the model will always generate the same result given a starting letter
            - to make the results more interesting, use np.random.choice to select a next letter that is likely, but not always the same
            - pick the next char’s index according to the prob distribution specified by y_hat<t+1>
            - this mean that if y_hat<t+1> = 0.16, pick the index i with 16% prob

                ```python
                np.random.seed(0)
                probs = np.array([0.1, 0.0, 0.7, 0.2])
                idx = np.random.choice(range(len(probs)), p = probs)
                ```

            - pick the index according to the distribution

            - note that the value that’s set to p should be set to a 1D vector
            - y_hat<t+1> is a 2D array
            - the first arg to np.random.choice is just an ordered list [0, 1, …, vocab_len-1]
            - not appropriate to use char_to_ix.values()
            - the order of values returned by a python dictionary .values() call will be the same order as they are added to the dictionary
            - np.ravel → takes a multi-dimensional array and returns its contents inside of a 1D vector

                ```python
                arr = np.array([[1,2],[3,4]])
                print("arr")
                print(arr)
                print("arr.ravel()")
                print(arr.ravel())

                '''
                arr
                [[1 2]
                [3 4]]
                arr.ravel()
                [1 2 3 4]
                '''
                ```

            - append is an in-place operation which means the changes made by the method will remain after the call completes
            - don’t do this

                ```python
                fun_hobbies = fun_hobbies.append('learning')  ## Doesn't give you what you want!
                ```

            - step 4: update to x<t>
                - the last step to implement in sample() is to update the variable
                - represent x<t+1> by creating a one-hot vector corresponding to the char for the prediction
                - then forward prop x<t+1> in step 1 and keep repeating the process until a “\n” char
            - in order to reset x before setting it to the new one-hot vector, set the values to zero

            ```python
            # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
            # GRADED FUNCTION: sample

            def sample(parameters, char_to_ix, seed):
                """
                Sample a sequence of characters according to a sequence of probability distributions output of the RNN

                Arguments:
                parameters -- Python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
                char_to_ix -- Python dictionary mapping each character to an index.
                seed -- Used for grading purposes. Do not worry about it.

                Returns:
                indices -- A list of length n containing the indices of the sampled characters.
                """

                # Retrieve parameters and relevant shapes from "parameters" dictionary
                Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
                vocab_size = by.shape[0]
                n_a = Waa.shape[1]

                ### START CODE HERE ###
                # Step 1: Create the a zero vector x that can be used as the one-hot vector 
                # Representing the first character (initializing the sequence generation). (≈1 line)
                x = np.zeros((vocab_size,1))
                # Step 1': Initialize a_prev as zeros (≈1 line)
                a_prev = np.zeros((n_a,1))

                # Create an empty list of indices. This is the list which will contain the list of indices of the characters to generate (≈1 line)
                indices = []

                # idx is the index of the one-hot vector x that is set to 1
                # All other positions in x are zero.
                # Initialize idx to -1
                idx = -1

                # Loop over time-steps t. At each time-step:
                # Sample a character from a probability distribution 
                # And append its index (`idx`) to the list "indices". 
                # You'll stop if you reach 50 characters 
                # (which should be very unlikely with a well-trained model).
                # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
                counter = 0
                newline_character = char_to_ix['\n']

                while (idx != newline_character and counter != 50):

                    # Step 2: Forward propagate x using the equations (1), (2) and (3)
                    a = np.tanh(Wax @ x + Waa @ a_prev + b)
                    z = Wya @ a + by
                    y = softmax(z)

                    # For grading purposes
                    np.random.seed(counter + seed) 

                    # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
                    # (see additional hints above)
                    idx = np.random.choice(range(vocab_size), p=y.ravel())

                    # Append the index to "indices"
                    indices.append(idx)

                    # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
                    # (see additional hints above)
                    x = np.zeros((vocab_size,1))
                    x[idx] = 1

                    # Update "a_prev" to be "a"
                    a_prev = a

                    # for grading purposes
                    seed += 1
                    counter +=1

                ### END CODE HERE ###

                if (counter == 50):
                    indices.append(char_to_ix['\n'])

                return indices
            ```

Building the Language Model

- Gradient Descent
    - implement a function performing one step of stochastic GD
    - going through training examples one at a time, so the optimization alg will be stochastic gradient descent
    - common optimization loop for an RNN
        - forward prop through the RNN to compute loss
        - backward prop through time to compute the grads of the loss w.r.t. the params
        - clip grads
        - update the params using gradient descent
    - Exercise 3 - optimize
        - implement the optimization process

            ```python
            def rnn_forward(X, Y, a_prev, parameters):
                """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
                It returns the loss' value as well as a "cache" storing values to be used in backpropagation."""
                ....
                return loss, cache

            def rnn_backward(X, Y, parameters, cache):
                """ Performs the backward propagation through time to compute the gradients of the loss with respect
                to the parameters. It returns also all the hidden states."""
                ...
                return gradients, a

            def update_parameters(parameters, gradients, learning_rate):
                """ Updates parameters using the Gradient Descent Update Rule."""
                ...
                return parameters
            ```

        - the weights and biases inside the parameters dictionary are being updated by the optimization, even though parameters is not on eof the returned values of the optimize function
        - the parameters dictionary is passed by reference into the function, so changes to this dictionary are making changes tot he parameters dictionary even when accessed outside of the function
        - python dictionaries and lists are pass by reference, which means that passing a dictionary into a function and modifying the dictionary within the function changes the same dictionary (not a copy of the dictionary)

        ```python
        # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: optimize

        def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
            """
            Execute one step of the optimization to train the model.

            Arguments:
            X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
            Y -- list of integers, exactly the same as X but shifted one index to the left.
            a_prev -- previous hidden state.
            parameters -- python dictionary containing:
                                Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                                Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                                Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                                b --  Bias, numpy array of shape (n_a, 1)
                                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
            learning_rate -- learning rate for the model.

            Returns:
            loss -- value of the loss function (cross-entropy)
            gradients -- python dictionary containing:
                                dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                                dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                                dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                                db -- Gradients of bias vector, of shape (n_a, 1)
                                dby -- Gradients of output bias vector, of shape (n_y, 1)
            a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
            """

            ### START CODE HERE ###

            # Forward propagate through time (≈1 line)
            loss, cache = rnn_forward(X, Y, a_prev, parameters)

            # Backpropagate through time (≈1 line)
            gradients, a = rnn_backward(X, Y, parameters, cache)

            # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
            gradients = clip(gradients, 5)

            # Update parameters (≈1 line)
            parameters = update_parameters(parameters, gradients, learning_rate)

            ### END CODE HERE ###

            return loss, gradients, a[len(X)-1]
        ```

- Training the Model
    - given the dataset of dinosaur names, use each line of the dataset as on training example
    - every 2k steps of stochastic GD, sample several randomly chosen names to see how the alg is doing
    - Exercise 4 - model
        - when examples[index] contains one dinosaur name (string), to create and example (X, Y)
        - set the index idx into the list of examples
            - using the for-loop, walk through the shuffled list of dinosaur names in the list examples
            - for example, if there are n_e examples, and the for-loop increments the index to n_e onwards, make the index cycle back to 0, to continue feeding the examples into the model when j is n_e, n_e + 1, etc
            - (n_e + 1) % n_e equals 1, which is otherwise the remainder when dividing (n_e + 1) by n_e
        - extract a single example from the list of examples
            - single_example: use the idx index that was previously set to get one word from the list of examples
        - convert a string into a list of characters: single_example_chars
            - single_example_chars: a string is a list of chars
            - use a list comprehension (recommended over for-loops) to generate a list of chars

                ```python
                str = 'I love learning'
                list_of_chars = [c for c in str]
                print(list_of_chars)

                # ['I', ' ', 'l', 'o', 'v', 'e', ' ', 'l', 'e', 'a', 'r', 'n', 'i', 'n', 'g']
                ```

        - convert list of chars to a list of ints: single_example_ix
            - create a list that contains the index numbers associated with each char
            - use the dictionary char_to_ix
            - combine this with the list comprehension that is used to get a list of chars from a string
        - create the list of input chars: X
            - rnn_forward uses the None value as a flag to set the input vector as a zero-vector
            - prepend the list [None] is front of the list of input char indices
            - there is more than one way to prepend a value to a list
            - one way is to add two lists together: [’a’] + [’b’]
        - get the integer representation of the newline char ix_newline
            - ix_newline: the newline char signals the end of the dinosaur name
            - get the integer representation of the newline char ‘\n’
            - use char_to_ix
        - set the list of labels (integer representation of the chars): Y
            - the goal is to train the RNN to predict the next letter in the name, so the labels are the list of chars that are one time-step ahead of the chars in the input X
                - Y[0] contains the same value as X[1]
            - the RNN should predict a newline at the last letter, so add ix_newline to the end of the labels
                - append the integer representation of the newline char to the end of Y
                - note that append is an in-place operation
                - might be easier to add two lists together

        ```python
        # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: model

        def model(data_x, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
            """
            Trains the model and generates dinosaur names. 

            Arguments:
            data_x -- text corpus, divided in words
            ix_to_char -- dictionary that maps the index to a character
            char_to_ix -- dictionary that maps a character to an index
            num_iterations -- number of iterations to train the model for
            n_a -- number of units of the RNN cell
            dino_names -- number of dinosaur names you want to sample at each iteration. 
            vocab_size -- number of unique characters found in the text (size of the vocabulary)

            Returns:
            parameters -- learned parameters
            """

            # Retrieve n_x and n_y from vocab_size
            n_x, n_y = vocab_size, vocab_size

            # Initialize parameters
            parameters = initialize_parameters(n_a, n_x, n_y)

            # Initialize loss (this is required because we want to smooth our loss)
            loss = get_initial_loss(vocab_size, dino_names)

            # Build list of all dinosaur names (training examples).
            examples = [x.strip() for x in data_x]

            # Shuffle list of all dinosaur names
            np.random.seed(0)
            np.random.shuffle(examples)

            # Initialize the hidden state of your LSTM
            a_prev = np.zeros((n_a, 1))

            # for grading purposes
            last_dino_name = "abc"

            # Optimization loop
            for j in range(num_iterations):

                ### START CODE HERE ###

                # Set the index `idx` (see instructions above)
                idx = j % len(examples)

                # Set the input X (see instructions above)
                single_example = examples[idx]
                single_example_chars = [c for c in single_example]
                single_example_ix = [char_to_ix[c] for c in single_example_chars]
                X = [None] + single_example_ix

                # Set the labels Y (see instructions above)
                ix_newline = char_to_ix['\n']
                Y = single_example_ix + [ix_newline]

                # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
                # Choose a learning rate of 0.01
                curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

                ### END CODE HERE ###

                # debug statements to aid in correctly forming X, Y
                if verbose and j in [0, len(examples) -1, len(examples)]:
                    print("j = " , j, "idx = ", idx,) 
                if verbose and j in [0]:
                    print("single_example =", single_example)
                    print("single_example_chars", single_example_chars)
                    print("single_example_ix", single_example_ix)
                    print(" X = ", X, "\n", "Y =       ", Y, "\n")

                # to keep the loss smooth.
                loss = smooth(loss, curr_loss)

                # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
                if j % 2000 == 0:

                    print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

                    # The number of dinosaur names to print
                    seed = 0
                    for name in range(dino_names):

                        # Sample indices and print them
                        sampled_indices = sample(parameters, char_to_ix, seed)
                        last_dino_name = get_sample(sampled_indices, ix_to_char)
                        print(last_dino_name.replace('\n', ''))

                        seed += 1  # To get the same result (for grading purposes), increment the seed by one. 

                    print('\n')

            return parameters, last_dino_name
        ```

Conclusion

- the alg started to generate plausible dinosaur names towards the end of training
- at first, it was generating random chars, but towards the end, the dinosaur names have some cool endings

Writing like Shakespeare

- a similar task to character-level text generation is generating Shakespearean poems
- instead of learning from a dataset of dinosaur names, use a collection of Shakespearean poems
- using LSTM cells, learn longer-term dependencies that span many characters in the text
- where a character appearing somewhere in a sequence can influence what should be a different character, much later int eh sequence
- these long-term dependencies were less important with dinosaur names
- packages

    ```python
    from __future__ import print_function
    from tensorflow.keras.callbacks import LambdaCallback
    from tensorflow.keras.models import Model, load_model, Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.utils import get_file
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from shakespeare_utils import *
    import sys
    import io
    ```

- the model has already been trained for 1000 epochs on a collection of Shakespearean poems call “The Sonnets”
- train the model for one more epoch
- when it finishes training for an epoch, run generate_output, which will prompt for an input
- the poem will start with the given sentence, and the RNN Shakespeare will complete the rest of the poem

    ```python
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
    ```

    ```python
    # Run this cell to try with different inputs without having to re-train the model 
    generate_output()

    '''
    Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: If thou is my love, thou is everything.

    Here is your poem: 

    If thou is my love, thou is everything.
    his summerargles doml constare behhed,
    on hiw styou thy all by being all whet stows, mins a falle;
    and theighs eyes, ksencst my to herp here which mady,
    dening these bawch that in my bears send,
    thy hath mesherce usrever ride faulds wo delest theng in,
    and what it me thy beauty he swill i dith,
    and cap with this semer's ast to time.
    low, nefred mans noel of that roans beer.
    which bouth bepay the 
    '''
    ```