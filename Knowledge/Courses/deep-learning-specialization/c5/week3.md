Basic Models

- sequence to sequence models
- useful from machine translation to speech recognition
- Sequence to sequence model
    - translating from french to english
    - input and output sequence
    - encoder network → RNN fed in french words
    - outputs vector that represents the input sentence
    - decoder network → takes input of encoder output
    - outputs the english sequence sequentially

- Image captioning
    - conv net to learn encoding / set of features for the image
    - output fed into RNN to create caption

Picking the Most Likely Sentence

- Machine translation as building a conditional language model
    - language model

    - estimates probability of a sentence
    - machine translation

    - decoder look identical to the language model
    - “conditional language model”
    - instead of modeling the probability of any sentence it is modeling the probability of the english translation conditioned on some input french sentence

- Finding the most likely translation
    - do not sample words at random

    - maximize conditional probability
- Why not a greedy search?
    - just take the most likely word at every time step
    - want to pick the entire sequence of words that maximized the joint probability
    - better to use an approximate search alg

Beam Search

- Beam search alg
    - step 1
        - tries to pick the first word
        - first step uses a network fragment
        - considers multiple alternatives
        - beam width → consider three outputs at a time

    - step 2
        - for each choice, consider what the second word might be
        - for y_hat<1> is set, looking for y_hat<2>
        - looking for pair that is most likely
        - probability of first word times the probability of the second word

        - cut the resulting choices to the beam width again
        - can reject first word possibilities altogether if the output three best probabilities are not within
        - three copies of the network are instantiated, if beam width is three

Refinements to Beam Search

- Length normalization
    - arg maxing probabilities of each word given the others
    - these probs are often much less than one
    - multiplying these numbers can result in numerical underflow
    - instead, maximize the log of the product

    - a very long sentence will have a very low prob
    - unnaturally tends to prefer very short translations as a result
    - can normalize by the number of words in the translation

    - keep track of top sentences, go through norm log prob vector
- Beam search discussion
    - large B: better result, slower
    - small B: worse result, faster
    - in production systems, beam width of around 10
    - unlike exact search algorithms like BFS (Breadth first search) or DFS (Depth first search), Beam Search runs faster but is not guaranteed to find the max for arg max P(y|x)

Error Analysis in Beam Search

- beam search is a heuristic search alg → approximate
- a mistake could be the RNN or the beam search
- RNN computes P(y|x)
- compute P(y*|x) and P(y_hat|x) → true vs predicted
- Error analysis on beam search
    - Case 1
        - P(y*|x) > P(y_hat|x)
        - beam search chose y_hat, but y* attains higher P(y|x)
        - conclusion → beam search is at fault
    - Case 2
        - P(y*|x) ≤ P(y_hat|x)
        - y* is a better translation than y_hat, but the RNN predicted otherwise
        - conclusion → RNN model is at fault
- Error analysis process
    - go through dev set and go through the mistakes
    - figures out what fraction of errors are due to beam search vs RNN model

Bleu Score

- could be multiple equally good translations
- how to evaluate
- Evaluating machine translation
    - given a machine translation, the bleu score measure how good the translation is
    - high bleu score if close to any of the references provided by human
    - Bleu → bilingual evaluation, understudy
    - one way to measure is to see if each word appears in the reference
    - the seven times → each word appears in both references
    - precision is not a good measure therefore
    - modified precision → give credit up to max number of appearances in the reference sentences

- Bleu score on bigrams (pairs of words)
    - enumerate possible sequential word pairs
    - count appearances

- Bleu score on unigrams
    - modified precision

    - the one reference to unigrams
    - for n-grams

- Bleu details
    - p_n = Bleu score on n-grams only
    - compute p1, p2, p3, p4, by convention
    - BP stands for brevity penalty

    - combined Bleu score

Attention Model Intuition

- The problem of long sequences
    - problem is asking model to memorize a whole long sentence and store in the encoder activations
    - instead, translate as you go, as would a human
    - bleu score decreases over sentence length
- originally developed for machine translation
- spread to other use cases
- using a BRNN, each of the position will compute a rich set of features
- S is denoted as the hidden state, to not confuse with the activations of the BRNN
- to generate the first part of the translation, which part of the french sentence should by the focus → primarily the first part of the sentence
- the model compute a set of attention weights
- alpha<1,1> is a weight for the outputs of the time steps of the BRNN
- next time step of the attention model will have a new set of attention weights

- attention depends on the activations at time t and the state from the previous time step

Attention Model

- allows the model to pay attention to only part of the input sentence while generating a translation
- a<t’> is the feature vector for time step t’
- t’ is french index
- separate forward only RNN
- time steps have input c for context → alpha params from the timesteps of the BRNN
- context will be defined as the weighted sum of the features from the different time steps weighted by the attention weights

- Computing attention a<t,t’>
    - amount of attention y<t> should pay to a<t’>
    - e vector is computed with a small NN from previous time step state and attention from time step t’

    - downside is speed → quadratic time
- Attention examples
    - input date to formatted date
    - visualization of a<t, t’>

Speech Recognition

- given audio clip x → to text transcript y
- audio is changes in air pressure
- common pre-processing step for audio data is to run the raw audio clip and generate a spectrogram
- was previously represented as phonemes → linguists
- with end to end deep learning, larger datasets allow for direct translation
- Attention model speech recognition
    - time frames of audio transcript

- CTC cost for speech recognition
    - connectionist temporal classification
    - in speech recognition, usually the number of input time steps is much bigger than the number of output time steps
    - Hz are per thousand
    - CTC allow for special output to collapse repeated characters not separated by ‘blank’ character

Trigger Word Detection

- alexa, siri, okay google
- still evolving
- set target labels to zero before the trigger word, one after
- could set a few one time steps