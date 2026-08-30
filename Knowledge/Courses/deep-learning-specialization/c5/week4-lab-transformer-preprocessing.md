Packages

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

Positional Encoding

- positional encoding formulas

- standard practice in natural language processing tasks to convert sentences into tokens before feeding texts into a language model
- each token is then converted into a numerical vector of fixed length called an embedding, which captures the meaning of the words
- in the transformer architecture, a positional encoding vector is added to the embedding to pass positional information throughout the model
- the meaning of the vectors can be difficult to grasp solely by examining the numerical representations, but visualizations can help give some intuition as to the semantic and positional similarity of the words
- when embeddings are reduced to two dimensions and plotted, semantically similar words appear together, while dissimilar words are plotted farther apart
- a similar exercise can be performed with positional encoding vectors - words that are closer in a sentence should appear closer when plotted on a cartesian plane, and when farther in a sentence, should appear father on the plane
- create a series of visualizations of word embeddings and positional encoding vectors to gain intuition into how positional encodings affect word embeddings and help transport sequential information through the transformer architecture
- Positional encoding visualizations
    - the positional_encoding function below

        ```python
        def positional_encoding(positions, d):
            """
            Precomputes a matrix with all the positional encodings 

            Arguments:
                positions (int) -- Maximum number of positions to be encoded 
                d (int) -- Encoding size 

            Returns:
                pos_encoding -- (1, position, d_model) A matrix with the positional encodings
            """

            # initialize a matrix angle_rads of all the angles 
            angle_rads = np.arange(positions)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d)[np.newaxis, :]//2)) / np.float32(d))
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

            pos_encoding = angle_rads[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)
        ```

    - define the embedding dimension as 100
    - this value must match the dimensionality of the word embedding
    - in the original paper, embedding sizes range from 100 to 1024, depending on the task
    - the authors also use a maximum sequence length ranging from 40 to 512 depending on the task
    - define the maximum sequence length to be 100, and the maximum number of words to be 64

        ```python
        EMBEDDING_DIM = 100
        MAX_SEQUENCE_LENGTH = 100
        MAX_NB_WORDS = 64
        pos_encoding = positional_encoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

        plt.pcolormesh(pos_encoding[0], cmap='RdBu')
        plt.xlabel('d')
        plt.xlim((0, EMBEDDING_DIM))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()
        ```

    - the norm of each of the vectors is always a constant
    - no matter what the value of pos is, the norm will always be the same value, which in this case is 7.071068
    - this property concludes that the dot product of two positional encoding vectors is not affected by the scale of the vector, which has important implications for correlation calculations

        ```python
        pos = 34
        tf.norm(pos_encoding[0,pos,:])

        # <tf.Tensor: shape=(), dtype=float32, numpy=7.071068>
        ```

    - another interesting property is that the norm of the difference between 2 vectors separated by k positions is also constant
    - keeping k constant and change pos, the difference will be of approximately the same value
    - this property is important because it demonstrates that the difference does not depend on the positions of each encoding, but rather the relative separation between encodings
    - being able to express positional encodings as linear function of one another can help the model to learn by focusing on the relative positions of words
    - this reflection of the difference in the positions of words with vector encodings is difficult to achieve, especially given that the values of the vector encodings must remain small enough so that they do not distort the word embeddings

        ```python
        pos = 70
        k = 2
        print(tf.norm(pos_encoding[0,pos,:] -  pos_encoding[0,pos + k,:]))

        # tf.Tensor(3.2668781, shape=(), dtype=float32)
        ```

- Comparing positional encodings
    - Correlation
        - the positional encoding matrix help to visualize how each vector is unique for every position
        - however, it is still not clear how these vectors can represent the relative position of the words in a sentence
        - to illustrate, calculate the correlation between pairs of vectors at every single position
        - a successful positional encoder will produce a perfectly symmetric matrix in which maximum values are located at the main diagonal - vectors in similar positions should have the highest correlation
        - the correlation values should get smaller as they move away from the main diagonal

            ```python
            # Positional encoding correlation
            corr = tf.matmul(pos_encoding, pos_encoding, transpose_b=True).numpy()[0]
            plt.pcolormesh(corr, cmap='RdBu')
            plt.xlabel('Position')
            plt.xlim((0, MAX_SEQUENCE_LENGTH))
            plt.ylabel('Position')
            plt.colorbar()
            plt.show()
            ```

    - Euclidean distance
        - also can use the euclidean distance instead of the correlation for comparing the positional encoding vectors
        - in this case, the visualization will display a matrix in which the main diagonal is 0, and its off-diagonal values increase as they move away from the main diagonal

            ```python
            # Positional encoding euclidean distance
            eu = np.zeros((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
            print(eu.shape)
            for a in range(MAX_SEQUENCE_LENGTH):
                for b in range(a + 1, MAX_SEQUENCE_LENGTH):
                    eu[a, b] = tf.norm(tf.math.subtract(pos_encoding[0, a], pos_encoding[0, b]))
                    eu[b, a] = eu[a, b]

            plt.pcolormesh(eu, cmap='RdBu')
            plt.xlabel('Position')
            plt.xlim((0, MAX_SEQUENCE_LENGTH))
            plt.ylabel('Position')
            plt.colorbar()
            plt.show()
            ```

Semantic embedding

- gain an intuition as to how positional encodings affect word embeddings by visualizing the sum of the vectors
- Load pretrained embedding
    - to combine a pretrained word embedding with the positional encodings created, start by loading one of the pretrained embeddings from the glove project
    - use the embedding with 100 features

        ```python
        embeddings_index = {}
        GLOVE_DIR = "glove"
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        print('d_model:', embeddings_index['hi'].shape)

        '''
        Found 400000 word vectors.
        d_model: (100,)
        '''
        ```

    - the embedding is composed of 400,000 words and each word embedding has 100 features
    - consider the following text that only contains two sentences
    - these sentences have no meaning
    - instead, the sentences are engineered such that
        - each sentence is composed of sets of words, which have some semantic similarities among each groups
        - in the first sentence similar terms are consecutive, while in the second sentence, the order is random
    - apply the tokenization to the raw text
        - feeding an array of plain text of different sentence lengths, it will produce a matrix with one row for each sentence, each of them represented by an array of size MAX_SEQUENCE_LENGTH
        - each value in the array represents each word of the sentence using its correspondign index in a dictionary (word_index)
        - the sequences shorter than the MAX_SEQUENCE_LENGTH are padded with zeros to create uniform length

        ```python
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

        print(data.shape)

        print(data)

        '''
        Found 11 unique tokens.
        (2, 100)
        [[ 1  2  3  4  5  6  7  8  9 10 11  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0]
         [ 3  2 11  8 10  5  4  7  1  9  6  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           0  0  0  0]]
        '''
        ```

    - to simplify the model, only need to obtain the embeddings for the different words that appear in the text that is being examined
    - in this case, filter out only the 11 words appearing in the sentences
    - the first vector will be an array of zeros and will codify all the unknown words

        ```python
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        print(embedding_matrix.shape)

        # (12, 100)
        ```

    - create an embedding layer using the weights extracted from the pretrained glove embeddings

        ```python
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                    trainable=False)
        ```

    - transform the input tokenized data to the embedding using the previous layer
    - check the shape of the embedding to make sure the last dimension of this matrix contains the embeddings of the words in the sentence

        ```python
        embedding = embedding_layer(data)
        print(embedding.shape)

        # (2, 100, 100)
        ```

- Visualization on a Cartesian plane
    - create a function to visualize the encoding of the words in a cartesian plane
    - use PCA to reduce the 100 features of the glove embedding to only two components

        ```python
        from sklearn.decomposition import PCA

        def plot_words(embedding, sequences, sentence):
            pca = PCA(n_components=2)
            X_pca_train = pca.fit_transform(embedding[sentence,0:len(sequences[sentence]),:])

            fig, ax = plt.subplots(figsize=(12, 6)) 
            plt.rcParams['font.size'] = '12'
            ax.scatter(X_pca_train[:, 0], X_pca_train[:, 1])
            words = list(word_index.keys())
            for i, index in enumerate(sequences[sentence]):
                ax.annotate(words[index-1], (X_pca_train[i, 0], X_pca_train[i, 1]))

        ```

    - plot the embedding of each of the sentences
    - each plot should display the embeddings of the different words

        ```python
        plot_words(embedding, sequences, 0)
        ```

    - plot the word of embeddings of the second sentence
    - note, the second sentence contains the same words that are in the first sentence, just in a different order
    - see that the order of the words does not affect the vector representants

Semantic and positional embedding

- combine the original glove embedding with the positional encoding
- use a 1 to 1 weight ratio between the semantic and the positional embedding

    ```python
    embedding2 = embedding * 1.0 + pos_encoding[:,:,:] * 1.0

    plot_words(embedding2, sequences, 0)
    plot_words(embedding2, sequences, 1)
    ```

- both plots changed drastically compared to their original counterparts
- the second image, which corresponds to the sentence in which similar words are not together, very dissimilar words such as red and wolf appear more close
- use different relative weights and see how this strongly impacts the vector representation of the words in the sentence

    ```python
    W1 = 1 # Change me
    W2 = 10 # Change me
    embedding2 = embedding * W1 + pos_encoding[:,:,:] * W2
    plot_words(embedding2, sequences, 0)
    plot_words(embedding2, sequences, 1)

    # For reference
    #['king queen man woman dog wolf football basketball red green yellow',
    # 'man queen yellow basketball green dog  woman football  king red wolf']
    ```

- setting W1 = 1 and W2 = 10, see how the arrangement of the words begins to take on a clockwise or anti-clockwise order depending on the positions of the words in the sentence
- under these params, the positional encoding vectors have dominated the embedding
- inverting the weights to W1 = 10, and W2 = 1, the plot resembles the original embedding visualizations and there are only a few changes between the positions of the plotted words
- in transformers, the word embedding is multiplied by sqrt(EMBEDDING_DIM)