Using Word Vectors to Improve Emoji Lookups

- in many emoji interfaces, the heart symbol and love symbol are distinct
- more flexible emoji interface can be made by using word vectors
- when using word vectors, even if the training set explicitly relates only a few words to a particular emoji, the alg will be able to generalize and associate additional words in the test set to the same emoji
- this works even if those additional words don’t even appear in the training set
- allows for an accurate classifier mapping from sentences to emojis, even using a small training set

Packages

```python
import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
from test_utils import *

%matplotlib inline
```

Baseline Model: Emojifier-V1

- Dataset EMOJISET
    - build a simple baseline classifier
    - tiny dataset (X, Y) where:
        - X contains 127 sentences (strings)
        - Y contains an integer label between 0 and 4 corresponding to an emoji for each sentence

    - load the dataset
    - the dataset is split between training (127 examples) and testing (56 examples)

        ```python
        X_train, Y_train = read_csv('data/train_emoji.csv')
        X_test, Y_test = read_csv('data/tesss.csv')
        ```

    - find the sentence with the max number of words, and will store it’s length maxLen (the number of words in the longest sentence, which will be used further)
        - split() breaks a string into a list of its words, so if x is a string, then len(x.split()) return the number of words in that string
        - the max function is an iterable and has a key argument, that can be used to modify the basis on which the largest element in the iterable is chosen
    - key has been chosen as the number of words in a string
    - so the max function will return the string with the largest number of words

        ```python
        maxLen = len(max(X_train, key=lambda x: len(x.split())).split())
        ```

    - print sentences from X_train and corresponding labels from Y_train
        - change idx to see different examples

        ```python
        for idx in range(10):
            print(X_train[idx], label_to_emoji(Y_train[idx]))

        '''
        never talk to me again 😞
        I am proud of your achievements 😄
        It is the worst day in my life 😞
        Miss you so much ❤️
        food is life 🍴
        I love you mum ❤️
        Stop saying bullshit 😞
        congratulations on your acceptance 😄
        The assignment is too long  😞
        I want to go play ⚾
        '''
        ```

- Overview of the Emojifier-V1
    - implement Emojifier-v1

    - inputs and outputs
        - the input of the model is a string corresponding to a sentence
        - the output will be a probability vector of shape (1, 5), (indicating that there are five emojis to choose from)
        - the (1, 5) probability vector is passed to an argmax layer, which extracts the index of the emoji with the highest probability
    - One-hot Encoding
        - to get the labels into a format suitable for training a softmax classifier, convert Y from its current shape (m, q) into a one-hot representation (m, 5)
            - each row is a one-hot vector giving the label of one example
            - here, Y-oh stands for “Y-one-hot” in teh variable names Y_oh_train and Y_oh_test

            ```python
            Y_oh_train = convert_to_one_hot(Y_train, C = 5)
            Y_oh_test = convert_to_one_hot(Y_test, C = 5)
            ```

        - see what convert_to_one_hot() did

            ```python
            idx = 50
            print(f"Sentence '{X_train[idx]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}", )
            print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")

            '''
            Sentence 'I missed you' has label index 0, which is emoji ❤️
            Label index 0 in one-hot encoding format is [1. 0. 0. 0. 0.]
            '''
            ```

- Implementing Emojifier-V1
    - convert each word in the input sentence into their word vector representations
    - take an average of the word vectors
    - use pre-trained 50D GloVe embeddings

        ```python
        word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
        ```

    - word_to_index: dictionary mapping from words to their indices in the vocabulary
        - 400,001 words, with the valid indices ranging from 0 to 400,000
    - index_to_word: dictionary mapping from indices to their corresponding words in the vocabulary
    - word_to_vec_map: dictionary mapping words to their GloVe vector representation

        ```python
        word = "cucumber"
        idx = 289846
        print("the index of", word, "in the vocabulary is", word_to_index[word])
        print("the", str(idx) + "th word in the vocabulary is", index_to_word[idx])

        '''
        the index of cucumber in the vocabulary is 113317
        the 289846th word in the vocabulary is potatos
        '''
        ```

    - Exercise 1 - sentence_to_avg
        - convert every sentence to lower-case, then split the sentence into a list of words
        - for each word in the sentence, access its GloVe representation
            - then take the average of all of the word vectors
        - the avg array of zeros should be a vector of the same shape as the other word vectors in the word_to_vec_map
            - choose a word that exists int eh word_to_vec_map and access its .shape filed
            - do not hard-code the word that is accessed
            - don’t assume that the word ‘the’ in the word_to_vec_map will be in the word_to_vec_map when the function is called later
        - any word vector can be used that is retrieved from the input sentence to find the shape of a word vector

        ```python
        # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: sentence_to_avg

        def sentence_to_avg(sentence, word_to_vec_map):
            """
            Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
            and averages its value into a single vector encoding the meaning of the sentence.

            Arguments:
            sentence -- string, one training example from X
            word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

            Returns:
            avg -- average vector encoding information about the sentence, numpy-array of shape (J,), where J can be any number
            """
            # Get a valid word contained in the word_to_vec_map. 
            any_word = next(iter(word_to_vec_map.keys()))

            ### START CODE HERE ###
            # Step 1: Split sentence into list of lower case words (≈ 1 line)
            words = sentence.lower().split()

            # Initialize the average word vector, should have the same shape as your word vectors.
            # Use `np.zeros` and pass in the argument of any word's word 2 vec's shape
            avg = np.zeros(word_to_vec_map[any_word].shape)

            # Initialize count to 0
            count = 0

            # Step 2: average the word vectors. You can loop over the words in the list "words".
            for w in words:
                # Check that word exists in word_to_vec_map
                if w in word_to_vec_map:
                    avg += word_to_vec_map[w]
                    # Increment count
                    count +=1

            if count > 0:
                # Get the average. But only if count > 0
                avg /= count

            ### END CODE HERE ###

            return avg
        ```

- Implement the Model
    - pass the average through froward prop
    - compute the cost
    - backprop to update the softmax params
    - Exercise 2 - model
        - the equations to implement in the forward pass and to compute the cross entropy cost
        - the variable Y_oh is the one-hot encoding of the output labels

        - possible to come up with a more efficient vectorized implementation

        ```python
        # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: model

        def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
            """
            Model to train word vector representations in numpy.

            Arguments:
            X -- input data, numpy array of sentences as strings, of shape (m,)
            Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
            word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
            learning_rate -- learning_rate for the stochastic gradient descent algorithm
            num_iterations -- number of iterations

            Returns:
            pred -- vector of predictions, numpy-array of shape (m, 1)
            W -- weight matrix of the softmax layer, of shape (n_y, n_h)
            b -- bias of the softmax layer, of shape (n_y,)
            """

            # Get a valid word contained in the word_to_vec_map 
            any_word = next(iter(word_to_vec_map.keys()))

            # Define number of training examples
            m = Y.shape[0]                             # number of training examples
            n_y = len(np.unique(Y))                    # number of classes  
            n_h = word_to_vec_map[any_word].shape[0]   # dimensions of the GloVe vectors 

            # Initialize parameters using Xavier initialization
            W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
            b = np.zeros((n_y,))

            # Convert Y to Y_onehot with n_y classes
            Y_oh = convert_to_one_hot(Y, C = n_y) 

            # Optimization loop
            for t in range(num_iterations): # Loop over the number of iterations

                cost = 0
                dW = 0
                db = 0

                for i in range(m):          # Loop over the training examples

                    ### START CODE HERE ### (≈ 4 lines of code)
                    # Average the word vectors of the words from the i'th training example
                    # Use 'sentence_to_avg' you implemented above for this  
                    avg = sentence_to_avg(X[i], word_to_vec_map)

                    # Forward propagate the avg through the softmax layer. 
                    # You can use np.dot() to perform the multiplication.
                    z = W @ avg + b
                    a = softmax(z)

                    # Add the cost using the i'th training label's one hot representation and "A" (the output of the softmax)
                    cost -= np.sum(Y_oh[i] * np.log(a))
                    ### END CODE HERE ###

                    # Compute gradients 
                    dz = a - Y_oh[i]
                    dW += np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
                    db += dz

                    # Update parameters with Stochastic Gradient Descent
                    W = W - learning_rate * dW
                    b = b - learning_rate * db

                assert type(cost) == np.float64, "Incorrect implementation of cost"
                assert cost.shape == (), "Incorrect implementation of cost"

                if t % 100 == 0:
                    print("Epoch: " + str(t) + " --- cost = " + str(cost))
                    pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py

            return pred, W, b
        ```

    - train the model and learn the softmax params (W, b)

        ```python
        np.random.seed(1)
        pred, W, b = model(X_train, Y_train, word_to_vec_map)
        # print(pred)

        '''
        Epoch: 0 --- cost = 410.4336578831472
        Accuracy: 0.5454545454545454
        Epoch: 100 --- cost = 63.612639746961435
        Accuracy: 0.9318181818181818
        Epoch: 200 --- cost = 0.7391301193275178
        Accuracy: 1.0
        Epoch: 300 --- cost = 0.3104825413333956
        Accuracy: 1.0
        '''
        ```

- Examining Test Set Performance
    - predict function

        ```python
        print("Training set:")
        pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
        print('Test set:')
        pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

        '''
        Training set:
        Accuracy: 1.0
        Test set:
        Accuracy: 0.9107142857142857
        '''
        ```

    - random guessing would lead to 20% accuracy, given that there are five classes
    - pretty good performance after only training on 127 examples
    - the model matches emojis to relevant words
        - in the training set, the alg saw the sentence “i love you” and level with a heart
        - check that the word treasure does not appear in the training set
        - nonetheless, what happens with “i treasure you”

        ```python
        X_my_sentences = np.array(["i treasure you", "i love you", "funny lol", "lets play with a ball", "food is ready", "today is not good"])
        Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

        pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
        print_predictions(X_my_sentences, pred)

        '''
        Accuracy: 0.8333333333333334

        i treasure you ❤️
        i love you ❤️
        funny lol 😄
        lets play with a ball ⚾
        food is ready 🍴
        today is not good 😄
        '''
        ```

    - because treasure has a similar embedding as love, the alg has generalized correctly even to a word it has never seen before
    - words such as heart, dear, beloved, or adore have embedding vectors similar to love
    - Word ordering isn’t considered in this model
        - note that the model doesn’t get the following sentence correct: “today is not good”
        - this alg ignores word ordering, so it’s not good at understanding phrases like not good
    - Confusion matrix
        - printing the confusion matrix can also help understand which classes are more difficult for the model
        - a confusion matrix shoes how often an example whose label is one class is mislabeled by the alg with a different class

        ```python
        # START SKIP FOR GRADING
        print(Y_test.shape)
        print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
        print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
        plot_confusion_matrix(Y_test, pred_test)
        # END SKIP FOR GRADING
        ```

Emojifier-V2: Using LSMTs in Keras

- build an LSTM model that takes word sequences as input
- this model will be able to account for word ordering
- still use pre-trained word embeddings to represent words
- feed word embeddings into an LSTM, and the LSTM will learn to predict the most appropriate emoji
- Packages

    ```python
    import numpy as np
    import tensorflow
    np.random.seed(0)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.initializers import glorot_uniform
    np.random.seed(1)
    ```

- Model Overview
    - implement the following

- Keras and Mini-batching
    - train Keras using mini-batches, most deep learning frameworks require that all sequences in the same mini-batch have the same length
    - this is what allows vectorization to work: a three-word sentence and a four-word sentence, then the computations needed for them are difference (one takes three steps of an LSTM, the other four) so it’s just not possible to do them both at the same time
    - Padding Handles sequences of varying length
        - the common solution to handling sequences of difference length is to use padding
        - set a maximum sequence length
        - pad all sequences to have the same length
    - example of padding
        - given a max sequence length of 20, pad every sentence with zeros so that each input sentence is of length 20
        - thus, the sentence “i love you” would be represented as (e_i, e_love, e_you, 0, …, 0)
        - any sentences longer than 20 words would have to be truncated
        - one way to choose the max sequence length is to just pick the length of the longest sentence in the training set

```python
The Embedding Layer
```

- the embedding matrix is represented as a “layer”
- the embedding matrix maps word indices to embedding vectors
    - the word indices are positive integers
    - the embedding vectors are dense vectors of fixed size
    - a dense vector is the opposite of a sparse vector
    - it means that most of its values are non-zero
    - as a counter-example, a one-hot encoded vector is not dense
- the embedding matrix can be derived in two ways
    - training a model to derive the embedding from scratch
    - using a pretrained embedding
- Using and updating pre-trained embeddings
    - create an embedding() layer in Keras
    - initialize the Embedding layer with GloVe 50D vectors
    - observe how Keras allows for trainable or fixed layers
    - because the training set is small, leave the GloVe embeddings fixed instead of updating them
- Inputs and outputs to the embedding layer
    - the embedding() layer’s input is an integer matrix of size (batch size, max input length)
        - this input corresponds to sentences converted into lists of indices
        - the largest int (highest word index) in the input should be no larger than the vocab size
    - the embedding layer outputs an array of shape (batch size, max input length, dimension of word vectors)
    - the figure shows the propagation of two example sentences through the embedding layer
        - both examples have been zero-padded to a length of max_len=5
        - the word embeddings are 50 units in length
        - the final dimension of the representation is (2, max_len, 50)

- Exercise 3 - sentences_to_indices
    - this function processes an array of sentences X and returns inputs to the embedding layer
    - convert each of the training sentences into a list of indices (the indices correspond to each word int eh sentence)
    - zero-pad all the lists so that the their length is the length of the longest sentence

    ```python
    for idx, val in enumerate(["I", "like", "learning"]):
        print(idx, val)
    ```

    ```python
    # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION: sentences_to_indices

    def sentences_to_indices(X, word_to_index, max_len):
        """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 

        Arguments:
        X -- array of sentences (strings), of shape (m,)
        word_to_index -- a dictionary containing the each word mapped to its index
        max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 

        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        """

        m = X.shape[0]                                   # number of training examples

        ### START CODE HERE ###
        # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
        X_indices = np.zeros((m, max_len))

        for i in range(m):                               # loop over training examples

            # Convert the ith training sentence to lower case and split it into words. You should get a list of words.
            sentence_words = X[i].lower().split()

            # Initialize j to 0
            j = 0

            # Loop over the words of sentence_words

            for w in sentence_words:
                # if w exists in the word_to_index dictionary
                if w in word_to_index:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.
                    X_indices[i, j] = word_to_index[w]
                    # Increment j to j + 1
                    j += 1

        ### END CODE HERE ###

        return X_indices
    ```

    ```python
    X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
    print("X1 =", X1)
    print("X1_indices =\n", X1_indices)

    '''
    X1 = ['funny lol' 'lets play baseball' 'food is ready for you']
    X1_indices =
     [[155345. 225122.      0.      0.      0.]
     [220930. 286375.  69714.      0.      0.]
     [151204. 192973. 302254. 151349. 394475.]]
    '''
    ```

- build embedding layer
    - the embedding layer takes as input a list of word indices
    - sentences_to_indices() create these word indices
    - the embedding layer will return the word embeddings for a sentence
- Exercise 4 - pretrained_embedding_layer
    - initialize the embedding matrix as a numpy array of zeros
        - the embedding matrix has a row for each unique word in the vocab
            - there is one additional row to handle unknown words
            - so vocab_size is the number of unique words plus one
        - each row will store the vector representation of one word
            - for example,. one row may be 50 positions long if using GloVe word vectors
    - fill in each row of the embedding matrix with the vector representation of a word
        - each word in word_to_index is a string
        - word_to_vec_map is a dictionary here the keys are strings and the values are the word vectors
    - define the keras embedding layer
        - use Embedding()
        - the input dimension is equal to the vocabulary length
        - the output dimension is equal to the number of positions in a word embedding
        - make the layer’s embeddings fixed
            - set trainable - True, then it will allow the optimization alg to modify the values of the word embeddings
            - n this case, don’t modify the word embeddings
    - set the embedding weights to be equal to the embedding matrix

    ```python
    # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION: pretrained_embedding_layer

    def pretrained_embedding_layer(word_to_vec_map, word_to_index):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

        Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """

        vocab_size = len(word_to_index) + 1              # adding 1 to fit Keras embedding (requirement)
        any_word = next(iter(word_to_vec_map.keys()))
        emb_dim = word_to_vec_map[any_word].shape[0]    # define dimensionality of your GloVe word vectors (= 50)

        ### START CODE HERE ###
        # Step 1
        # Initialize the embedding matrix as a numpy array of zeros.
        # See instructions above to choose the correct shape.
        emb_matrix = np.zeros((vocab_size, emb_dim))

        # Step 2
        # Set each row "idx" of the embedding matrix to be 
        # the word vector representation of the idx'th word of the vocabulary
        for word, idx in word_to_index.items():
            emb_matrix[idx, :] = word_to_vec_map[word]

        # Step 3
        # Define Keras embedding layer with the correct input and output sizes
        # Make it non-trainable.
        embedding_layer = Embedding(vocab_size, emb_dim)
        ### END CODE HERE ###

        # Step 4 (already done for you; please do not modify)
        # Build the embedding layer, it is required before setting the weights of the embedding layer. 
        embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.

        # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
        embedding_layer.set_weights([emb_matrix])

        return embedding_layer
    ```

    ```python
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][1] =", embedding_layer.get_weights()[0][1][1])
    print("Input_dim", embedding_layer.input_dim)
    print("Output_dim",embedding_layer.output_dim)

    '''
    weights[0][1][1] = 0.39031
    Input_dim 400001
    Output_dim 50
    '''
    ```

- Building the Emojifier-V2
    - the model will be fed the embedding layer’s output to an LSTM network

    - Exercise 5 - Emojify_V2
        - this function build a Keras graph of the architecture
        - the model takes as input an array of sentences of shape (m, max_len,) defined by input_shape
        - the model outputs a softmax probability vector of shape (m, C=5)
        - use the following Keras layers
            - Input()
                - set the shape and dtype params
                - the inputs are ints, so the data type can be specifies as a string
            - LSTM()
                - set the units and return_sequences params
            - Dropout()
                - set the rate param
            - Dense()
                - set the units
                - note that Dense() has an activation param
                - do not set the activation within Dense() here
            - Activation()
                - pass the activation as a lowercase string
            - Model()
                - set inputs and outputs
        - Keras layers return an object
        - the outputs of the previous layer are input args to that object
        - the returned object can be created and called in the same line

            ```python
            # How to use Keras layers in two lines of code
            dense_object = Dense(units = ...)
            X = dense_object(inputs)

            # How to use Keras layers in one line of code
            X = Dense(units = ...)(inputs)
            ```

        - the embedding_layer that is returned by pretrained_embedding_layer is a layer object that can be called as a function, passing in a single argument

            ```python
            raw_inputs = Input(shape=(maxLen,), dtype='int32')
            preprocessed_inputs = ... # some pre-processing
            X = LSTM(units = ..., return_sequences= ...)(processed_inputs)
            X = Dropout(rate = ..., )(X)
            ...
            X = Dense(units = ...)(X)
            X = Activation(...)(X)
            model = Model(inputs=..., outputs=...)
            ...
            ```

        ```python
        # UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: Emojify_V2

        def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
            """
            Function creating the Emojify-v2 model's graph.

            Arguments:
            input_shape -- shape of the input, usually (max_len,)
            word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
            word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

            Returns:
            model -- a model instance in Keras
            """

            ### START CODE HERE ###
            # Define sentence_indices as the input of the graph.
            # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
            sentence_indices = Input(shape=(input_shape), dtype='int32')

            # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
            embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

            # Propagate sentence_indices through your embedding layer
            # (See additional hints in the instructions).
            embeddings = embedding_layer(sentence_indices)   

            # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
            # The returned output should be a batch of sequences.
            X = LSTM(128, return_sequences=True)(embeddings)
            # Add dropout with a probability of 0.5
            X = Dropout(0.5)(X)
            # Propagate X trough another LSTM layer with 128-dimensional hidden state
            # The returned output should be a single hidden state, not a batch of sequences.
            X = LSTM(128)(X)
            # Add dropout with a probability of 0.5
            X = Dropout(0.5)(X)
            # Propagate X through a Dense layer with 5 units
            X = Dense(5)(X)
            # Add a softmax activation
            X = Activation("softmax")(X)

            # Create Model instance which converts sentence_indices into X.
            model = Model(sentence_indices, X)

            ### END CODE HERE ###

            return model
        ```

    - create the model and check the summary
        - because all sentences in the dataset are less than 10 words, max_len = 10
        - the architecture uses 20,223,927 params, of which 20,000,50 (the word embeddings) are non-trainable, with the remaining 223,877 being trainable
        - because the vocab size has 400,001 words (with valid indices from 0 to 400,000) there are 400,001 * 50 = 20,000,050 non-trainable params

        ```python
        model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
        model.summary()

        '''
        Model: "functional_3"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_2 (InputLayer)         [(None, 10)]              0         
        _________________________________________________________________
        embedding_3 (Embedding)      (None, 10, 50)            20000050  
        _________________________________________________________________
        lstm_2 (LSTM)                (None, 10, 128)           91648     
        _________________________________________________________________
        dropout_2 (Dropout)          (None, 10, 128)           0         
        _________________________________________________________________
        lstm_3 (LSTM)                (None, 128)               131584    
        _________________________________________________________________
        dropout_3 (Dropout)          (None, 128)               0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 5)                 645       
        _________________________________________________________________
        activation_1 (Activation)    (None, 5)                 0         
        =================================================================
        Total params: 20,223,927
        Trainable params: 20,223,927
        Non-trainable params: 0
        _________________________________________________________________
        '''
        ```

    - compile the model
        - after creating the model in Keras, it needs to be compiled and define what loss, optimizer and metrics to use
        - compile with categorical_crossentropy loss, adam optimizer and [’accuracy’] metrics

        ```python
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        ```

- Train the Model
    - Emojifier-V2 model takes as input an array of shape (m, max_len) and outputs probability vectors of shape (m, number of classes)
    - thus, convert X_train (array of sentences as strings) to X_traing_indices (array of sentences as list of word indices), and Y_train (labels as indices) to Y_train_oh (labels as one-hot vectors

        ```python
        X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
        Y_train_oh = convert_to_one_hot(Y_train, C = 5)
        ```

    - fit the Keras model on X_train_indices and Y_train_oh, using epochs = 50 and batch_size = 32

        ```python
        model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

        '''
        Epoch 1/50
        5/5 [==============================] - 2s 364ms/step - loss: 1.6164 - accuracy: 0.1894
        Epoch 2/50
        5/5 [==============================] - 2s 364ms/step - loss: 1.5266 - accuracy: 0.3333
        ...
        5/5 [==============================] - 2s 360ms/step - loss: 0.1053 - accuracy: 0.9545
        Epoch 50/50
        5/5 [==============================] - 2s 360ms/step - loss: 0.1794 - accuracy: 0.9470
        '''
        ```

    - evaluate the model

        ```python
        X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
        Y_test_oh = convert_to_one_hot(Y_test, C = 5)
        loss, acc = model.evaluate(X_test_indices, Y_test_oh)
        print()
        print("Test accuracy = ", acc)

        '''
        2/2 [==============================] - 0s 3ms/step - loss: 0.4581 - accuracy: 0.8929

        Test accuracy =  0.8928571343421936
        '''
        ```

    - see mislabeled examples

        ```python
        # This code allows you to see the mislabelled examples
        C = 5
        y_test_oh = np.eye(C)[Y_test.reshape(-1)]
        X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
        pred = model.predict(X_test_indices)
        for i in range(len(X_test)):
            x = X_test_indices
            num = np.argmax(pred[i])
            if(num != Y_test[i]):
                print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

        '''
        Expected emoji:🍴 prediction: any suggestions for dinner	😄
        Expected emoji:😄 prediction: you brighten my day	😞
        Expected emoji:😄 prediction: will you be my valentine	😞
        Expected emoji:😄 prediction: What you did was awesome	😞
        Expected emoji:😞 prediction: go away	⚾
        Expected emoji:❤️ prediction: family is all I have	🍴
        '''
        ```

    - try on sample sentence

        ```python
        # Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
        x_test = np.array(['I cannot play'])
        X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
        print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))

        # I cannot play ⚾
        ```

- LSTM version accounts for word order
    - the emojify-V1 model did not predict “not feeling happy” correctly, by V2 did
    - the current model still isn’t very robust at understanding negation
        - this is because the training set is small and doesn’t have a lot of examples of negation
        - if the training set were larger, the LSTM model would be much better at understanding more complex sentences