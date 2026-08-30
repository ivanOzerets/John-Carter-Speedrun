Word Representation

- word embeddings are a way of representing words
- have been using a vocab, for example of 10k
- used a one-hot vector
- alg has a tough time generalizing from orange … juice to apple … juice
- because the relationship between any two one hot vectors is zero
- Featurized representation: word embedding
    - instead learn a set of features and values

    - better representation than one-hot vectors
- Visualizing word embeddings
    - t-SNE → 300D to 2D for visualization purposes

Using Word Embeddings

- how to use representations of different words and plug them into NLP applications
- Named entity recognition example
    - feature representations can give correlation understanding to the model
    - take the learned word embedding on the large dataset and apply to the smaller data size task of name entity recognition

- Transfer learning and word embeddings
    - learn word embeddings from large text corpus
    - transfer embedding to new task with smaller training set
    - optional: continue to finetune the word embeddings with new data
- Relation to face encoding
    - encoding and embedding are referred to the same vectors

Properties of Word Embeddings

- word embedding can also help with analogy reasoning
- Analogies
    - hope the featurized representations of a set words look like this
    - can the alg figure it out automatically, man → woman, king → ?
    - subtracting embeddings results in similar comparisons, gender and gender between man and woman and king and queen

- Analogies using word vectors
    - vector differences are similar in like pairs of words
    - find a word w that maximizes similarity between (ew, eking - eman + ewoman)

    - t-SNE takes 300D data and maps in non-linear way into 2D
    - parallelogram relations are reliably more true in the higher dimensions space
- Cosine similarity
    - most commonly used similarity
    - inner product between u and v
    - can also use squared or Euclidean distance
    - difference is how it normalizes the lengths of the vectors u and v

Embedding Matrix

- each word will have its own embedding
- embedding matrix E
- E dot (word embedding) selects the vector corresponding to the embedding word

Learning Word Embeddings

- started with complex algs, but over time the algs got simpler and still got good results
- Neural language model
    - if trying to predict the next word
    - neural language model is a reasonable way to learn a set of embeddings
    - pipeline: one-hot vector of every word in the sentence doted with embedding matrix, all outputs fed into fully connected layer into softmax output of the dictionary size
    - more common to have a fixed historical window, next word given, say, previous four words

- Other context/target pairs
    - target word and context
    - context is up to the learning problem
        - last four words
        - four words left and right
        - last one word
        - nearby one word → skip gram

Word2Vec

- Skip-grams
    - randomly pick a word to be the content word
    - randomly pick another word within some window, plus minus five words, for examples
    - use supervised learning problem to learn good word embeddings
- Model
    - to represent the input, start with one hot vector of the context word
    - multiply with E which gives the embedding vector
    - e_c is fed to a node in softmax unit
    - theta_t is the parameter associated with output t

    - loss function is normal

    - called the skip gram model
    - E and softmax unit has parameters
    - taking one word as image and skips a few words to predict
- Problems with softmax classification
    - computational speed
    - need to carry out a sum over all vocab words
    - the denominator is to hard to compute

    - instead, use a hieratical softmax
    - one classifier to identify if the output is in the first half or second half
    - binary tree classifier
    - log scale rather than linear
    - how to sample the context c
        - plus minus ten, per se
        - sample uniformly random is a choice, but common words get sampled extremely frequently
        - e_c will be updated for the common words much more frequently
        - different heuristics are used to get common and uncommon context words

Negative Sampling

- Word2Vec downside is the softmax softmax computation
- Defining a new learning problem
    - given a pair of words, predict a context target pair
    - positive example: sample context word, look around for a target word
    - negative example: same context word, random target word from the dictionary
    - okay if the random negative target word appears nearby
    - alg has inputs of context and target word pairs and predicts target label zero or one

    - k = 5-20 for smaller datasets
    - k = 2-5 for larger dataset
- Model
    - theta for each possible target word
    - embedding vector for each possible context word
    - one-hot vector doted with embedding matrix gets the embedding vector fed into dictionary size amount of logistic regression outputs, but only train the ones that are selected as positive or negative examples
    - so instead of one giant 10k softmax → 10k binary classifications, each which is quite cheap, and each iterations only updates 5, for example, of them
    - called negative sampling
- Selecting negative examples
    - sample according to empirical frequency → the, of, and and have too high representation
    - uniformly random → not representative of the word distribution
    - heuristic value → between two extremes

GloVe Word Vectors (global vectors for word representation)

- previously sampling pairs of words, context and target words, by picking two words that appear in close proximity to each other in the text corpus
- let Xij = # times j appears in context of i
- depends on choice of context
- Model
    - how related are words i and j
    - solve for params theta and e with GD
    - extra weighting term f(Xij) = 0 if Xij = 0 to avoid negative infinity from the log
    - theta and e are symmetric
    - for a given word w, take the average

- A note on the featurization view of word embeddings
    - cannot guarantee that the individual components of the word embeddings are interpretable

Sentiment Classification

- might not have a huge labeled training set
- with word embeddings, possible to build good sentiment classifiers even with only modest-size labeled training sets
- Sentiment classification problem
    - one to five star review from comments

- Simple sentiment classification model
    - perform normal vector embedding to get classifiers
    - average or sum → 300D feature vector → softmax classifier

    - problem → ignores word order
    - sentences like the following are confusing

    - instead us RNN
- RNN for sentiment classification
    - embedding vectors are fed into an RNN

    - because the word embeddings can be trained from a much larger data set, this will do a better job generalizing to even new words

Debiasing Word Embeddings

- The problem of bias in word embeddings
    - man:woman as king:queen
    - man_computer_programmer as woman:homemaker X
    - gender stereotype
    - word embeddings can reflect gender, ethnicity, age, sexual orientation, and other biases of the text used to train the model
    - biases are reflective of text by people
- Addressing bias in word embeddings
    - Identify bias direction
    - average differences of between bias parts: he and she, male and female, ..
- Neutralize: for ever word that is not definitional, project to get rid of bias
    - project onto the non-bias direction to reduce or eliminate in the bias direction
- Equalize pairs
    - pairs of words difference should only be the gender, for example
    - grandmother/grandfather, girl/boy

    - training a classifier to determine which words are definitional results in that most words are not definitional