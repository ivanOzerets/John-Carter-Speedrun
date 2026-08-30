Transformer Network Intuition

- rnn have problem with vanishing gradients which make it hard to capture long range dependencies and sequences
- gru and lstm resolve the problems with the use of gates to control the flow of information
- came with increased complexity
- still sequential → ingested input one token at a time
- as if each unit was a bottleneck to the flow of information
- final output needs to compute all computation before
- transformers allow to run a lot more computations for an entire sequence in parallel
- ingest an entire sentence at the same time
- combining attention and cnn
- rnn → sequential
- cnn → a lot of inputs in parallel
- self-attention → five representation for five words, for example
- multi-head attention → for loop basically over self-attention

Self-Attention

- self-attention intuition
    - create attention-based representations for each of the words in the input sentence
    - A(q, K, V) → A<1>, …, A<5>, query, key, and value
    - could look up a word embedding, but what context behind the word is important

- associate each with three values → query, key, and value
- computed as learnt matrix
- compute inner product between query and keys of every word
- goal is to pull out the most useful information for the best representation A<3>, for example
- softmax the values
- then multiply by the value of the word and sum

- denominator is scaled dot product so it doesn’t explode

Multi-Head Attention

- head → self-attention for a sequence
- multiply q, k, and v matrices with weight matrices W^q, W^k, W^v
- five vectors for five words
- second head has new set of weight matrices
- first question of the weight matrices → what’s happening?
- second question → when?
- number of heads is represented with h
- each q, k, and v have a weight matrix and each head has a W^q, W^k, W^v matrix
- no one head’s value is determined by the value of any other head so can be computed in parallel

Transformer Network

- transformer
    - useful to add SOS (start of sentence) and EOS (end of sentence)
    - first step is to feed embeddings into an encoder block → multi-head attention layer
    - then fed into a feed forward NN
    - ran through encoder n times → usually six

    - decoder job is to output the english translation
    - <SOS> token gets fed in
    - first multi-head attention block is used to generate Q and the output of the encoder is used to generate K and V for the next multi-head attention block
    - input to decoder is whatever has been translated so far
    - second multi-head attention fed into feed forward NN
    - decoder is run n times → usually six

- transformer details
    - positional encoding of input
    - self-attention → nothing indicates the position of a word
    - so use combination of sin and cos equations
    - create a positional embedding vector of the same dimension as the word embeddings
    - pos is the numerical position
    - then positional embedding layer is composed of the following equations

    - these equations create an alternating vector for each index i

    - the positional embedding is added to the word embedding
    - the output of the encoding block contains contextual semantic embedding and positional information
    - the output is d → the max length of sequence the model can take
    - also pass the position encodings with residual connections → like resnet
    - to pass along positional information through the whole network
    - also uses add and norm to speed up learning
    - repeated throughout
    - after → linear → softmax
    - masked multi-head attention is important only during the training process where dataset has correct translations
    - because full correct output is available, generating words one at a time is unnecessary
    - instead, block out last part of the sentence to mimic test time or prediction
    - pretends the network has perfectly translated the first part of the word and sees if the network can predict the next word in the sequence accurately

    - improvements such as BERT or DistilBERT

### **Week 1:**

- [Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086) (GitHub: karpathy)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Andrej Karpathy blog, 2015)
- [deepjazz](https://github.com/jisungk/deepjazz) (GitHub: jisungk)
- [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf) (Gillick, Tang & Keller, 2010)
- [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf) (Keller & Morrison, 2007)
- [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf) (Pachet, 1999)

### **Week 2:**

- [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) (Bolukbasi, Chang, Zou, Saligrama & Kalai, 2016)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) (Pennington, Socher & Manning, 2014)
- [Woebot](https://woebothealth.com/).

### **Week 4:**

- [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing?) (by [DeepLearning.AI](https://www.deeplearning.ai/))
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, 2017)