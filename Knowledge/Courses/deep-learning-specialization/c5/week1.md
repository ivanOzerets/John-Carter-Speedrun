Why Sequence Models?

- Examples of sequence data
    - speech recognition → map input audio clip x and map to text transcript y
    - music generation → only output y is a sequence
    - sentiment classification → text to star count
    - DNA sequence analysis → which part of the DNA sequence corresponds to a protein
    - machine translation → language to language
    - video activity recognition → video to description
    - name entity recognition → identity people in sentence
- all examples can be address as supervised learning

Notation

- Motivating example
    - x: Harry Potter and Hermione Granger invented a new spell
    - y: one output per input word, is the word part of a name
    - sequence of nine word
    - nine sets of features to represent the nine words
    - x<1>, x<2>, …, x<t>, ..., x<9>
    - y<1>, y<2>, …
    - x(i)<t> is a word in a training example
    - Tx(i) is a length of input sequence i
- Representing words
    - vocabulary or dictionary → lists of words
    - [a, aaron, …, harry, …, potter, …, zulu]
    - size of 10k → quite small
    - 30k - 50k is common
    - build by using training set and get top 10k words
    - words are represented as a one-hot encoding vector
    - new word → unknown word <UNK>

Recurrent NN Model

- Why not a standard network?
    - problems with feeding one hot vectors into standard fc NN

    - inputs and outputs can be different lengths in different examples
    - doesn’t share features learned across different positions of text
    - also the first layer would have an enormous number of params because of 10k one-hot vectors
- Recurrent NN
    - feed first word into NN layer → try predict output
    - when feeding next word, the activation value from timestep one is fed in also
    - made up activation at time zero → usually vector of zeros
    - parameters are shared from one times to the next
    - weakness is the NN only uses information from earlier in the sequence
    - later addressed with bidirectional RNN

- Forward Prop
    - Wax mean multiplied by x like quantity to get an a like quantity
    - tanh is a common choice

- Simplified RNN notation
    - take both Wa’s, put them side by side and stack them horizontally
    - stack a<t-1>, x<t> → stack vector on top of each other

Backprop Through Time

- Forward prop and backprop
    - params Wa,ba are used for every computation
    - loss is computed for every time step

    - activation params are updated backwards through time

Different Types of RNNs

- Tx and Ty may not always be the same
- machine translation, for example
- Examples of RNN architectures
    - many to many architecture

    - sentiment classification, x → text, y → 0/1 or 1 - 5
    - many to one architecture

    - one to one arch
    - like most basic fc NN

    - one to many arch → music generation
    - input could be genre or null

    - another many to many arch, language translation

- Summary

Language Model and Sequence Generation

- rnn’s are good at natural language processing
- What is language modelling?
    - speech recognition
        - P(the apple and pair salad) = 3.2x10-13
        - P(the apple and pear salad) = 5.7x10-10
    - P(sentence) = ?
    - for language models, its useful to represent the sentences as output y rather input x
    - estimates particular sequence of words
- Language modelling with an RNN
    - training set: large corpus of english text
    - tokenize to for a vocabulary
    - also add an end of sentence token <EOS>
    - punctuation can be added as a word
    - unknown word has <UNK> token
- RNN model
    - a<1> tries to predict the probability of the first word out of any word
    - then a<2> gets the correct first word

    - loss is sum over all timesteps for the softmax loss function

    - change of the entire sentence

Sampling Novel Sequences

- informally getting a sense of what is learned
- Sampling a sequence from a trained RNN
    - sequence model models the chance of a particular sequence of words
    - sample the distribution to generate novel sequences
    - to sample
        - randomly sample according the softmax distribution output
        - np.random.choice according the the outputs distribution
        - sequence end when <EOS> is generated, or just sample a certain number
- Character-level language model
    - Vocabulary = a, b, .., z, _, ., , ;, 0, …, 9, A, … ,Z]
    - then sequence will be individual characters
    - no unknown tokens
    - disadvantage → longer sequences
    - not as good as capturing long range dependencies
- Sequence generation
    - depends on training data

Vanishing Gradient with RNNs

- the cat which, …, was full
- the cats, which, …, were full
- need long term dependencies
- later timestamp gradients have little effect on early computations
- value is influenced by close values
- vanishing gradients are more common than exploding gradients → NaN is usually seen
- for exploding gradients → use gradient clipping

Gated Recurrent Unit (GRU)

- helps against vanishing gradients
- RNN unit
    - rnn review

- GRU (simplified)
    - new unit c → memory cell
    - c<t> = a<t>, output unit
    - at every timestep, consider overwriting c with ~c<t>
    - update gate Gamma_u is a value between 0 and 1, actually 0 or 1 because of sigmoid
    - c is the candidate, Gamma decides

    - the actual value of c<t>

    - picture format explanation

    - because gamma can be close to zero, doesn’t suffer from vanishing gradient problem as much
    - c<t>, ~c<t>, and gamma_u are all the same shape
    - gamma_u tells the memory cell bits to update
- Full GRU
    - add relevance gamma parameters

Long Short Term Memory (LSTM)

- even more powerful
- GRU and LSTM
    - GRU

    - LSTM
        - two gates, update and forget
        - and new output gate

        - so long as forget and update gates are set appropriately, easy for LSTM to have a value c<0> be passed all the way down the timesteps
        - peephole connection → gate values also depend on c<t-1>, a variation
        - not every element of gate values can effect all other values, just the corresponding element, one-to-one
        - LSTM were first, GRU is simpler, easier to build a larger model
        - no standard on which one to pick

Bidirectional RNN

- at a timestep, take information from earlier and later in the sequence
- specify a backward recurrent layer
- defines an acyclic graph
- predictions made after both forward props

- for a lot of NLP problems, BRNN w/ LSTM blocks is common
- disadvantage is the whole sequence is needed beforehand

Deep RNNs

- more layers before making predictions
- three layers are already a lot
- also possible to have three recurrent followed by a deep network