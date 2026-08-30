What is a NN?

- Housing Price Prediction
    - based on nodes

    - ReLU function → Rectified Linear Unit
    - can be formed by one neuron

    - other features can include
        - price
            - family size
                - size
                - number of bedrooms
            - walkability
                - zip code (postal code)
            - school quality
                - zip code
                - wealth

    - each neuron can be a relu or some other non-linear function
    - middle features will be figured out on its own
    - all four input features will be inputs → let the NN decide

Supervised Learning with NN

- Supervised Learning
    - input x is mapped to output y
    - online advertising is very lucrative

    - real estate and online advertising → standard NN
    - photo tagging → CNN
    - speech recognition and machine translation → RNN
    - autonomous driving → custom/hybrid
- NN examples

- Supervised Learning
    - Structured Data

    - Unstructured Data

    - computers are now much better at interpreting unstructured data with NNs

Why is Deep Learning taking off?

- Scale drives deep learning progress
    - if you plot the performance of learning algorithms like support vector machines SVM or logistic regression as a function of the amount of data, performance plateaus
    - linear models struggle as data increases
    - NN perform better

    - amount of ‘labeled’ data is important
    - in small training set, the order of performance on models is not well defined
    - huge breakthroughs come through how to make NN run faster
    - for example, switching from sigmoid to ReLU
    - problem with sigmoid is that gradient decent because really slow as the gradient because nearly zero near the ends of the sigmoid function
    - the process of training NN are very iterative
    - idea → code → experiment

About this Course

- Intro
- Basics of NN programming
- One hidden layer NN
- Deep NN

Geoffrey Hinton Interview

- restricted Boltzmann machines
- derestricted Boltzmann machines
- sigmoid belief net
- variational methods
    - en alg
    - e step
- approximate paper
- residual networks
- recirculation algorithm
- autoencoder
- fast weights → adapt rapidly, but decay rapidly
- capsule → feature that has a lot of properties
- routine by agreement
- wegstein algorithm
- variational altering code → reparameterization tricks
- generative adversarial nets
- when you have a good idea and other people think it is rubbish, that is the sign of a really good idea
- rms algorithm