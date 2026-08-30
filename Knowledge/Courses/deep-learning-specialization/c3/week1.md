Why ML Strategy

- how to get machine learning systems working quickly and efficiently
- got 90% accuracy, but not good enough
- ideas
    - more data
    - more diverse training set
    - train algorithm longer with gradient descent
    - try adam instead of gradient descent
    - try bigger network
    - try smaller network
    - try dropout
    - add L2 regularization
    - network architecture
        - activation functions
        - number of hidden units
        - …
- quick and effective ways to know which of these are worth pursuing

Orthogonalization

- so many hypers to tune
- clear-eyed on what to tune to change what effect
- TV tuning example
    - tuning every knob on an old school tv so have a interpretable function per knob
    - in contrast, one knob that tunes every knob by some values → almost impossible to center the image
    - orthogonalization means to tune each knob to do only one thing

- Chain of assumptions in ML
    - fit training set well on cost function
        - bigger network
        - better optimization
        - …
    - comparable to human level performance often
    - fit dev set well on cost function
        - regularization
        - bigger training set
    - fit test set well on cost function
        - bigger dev set
    - performs well in real world
        - dev set or cost function
    - early stopping is less orthogonalized because it affects how well the training set it fit and to improved dev set performance

Single Number Evaluation Metric

- having a single real number evaluation metric makes progress much faster
- Using a single number evaluation metric
    - machine learning is heavily empirical
    - precision → of the positive examples, which percentage actually are positive
    - recall → of all positive example, what percentage were correctly recognized
    - often a trade off
    - not sure which classifier is better if one has better recall vs precision
    - instead define a new metric that combines precision and recall
    - f1 score = average of P and R
    - 2/((1/P) + (1/R)) → hormonic mean

    - measured on dev set
- Another example
    - difficult where there are many classifiers in many geographies
    - average across geographies

Satisficing and Optimizing Metric

- Another cat classification example
    - care about accuracy and running time
    - combine into one evaluation metric
    - cost = accuracy - 0.5 * running time

    - or maximize accuracy but run time ≤ 100 ms
    - N metrics: one is optimizing, N-1 satisficing with thresholds
    - with trigger words, ok google, maximize accuracy w/ 1 false positive every 24 hours

Train/Dev/Test Distributions

- Cat classification dev/test sets
    - development set, help out, cross validation set
    - dev and test should come from the same distribution
    - randomly shuffle data from all sources
- Guideline
    - choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on

Size of the Dev and Test Set

- Old way of splitting data
    - 70% training / 30% test
    - 60% training / 20% dev / 10% test
    - with a million data points, then 98% training, 1% and 1%
- Size of test set
    - set the test set to be big enough to give high confidence in the overall performance of the system
    - sometimes just a train and dev set is okay
    - not recommended however

When to Change Dev/Test sets and Metrics?

- if you put the target in the wrong place
- lower error is not always better if the model is letting though some unacceptable mistakes
- add a weight to the evaluation metric where the weight is really high if the mistake is made

- Orthogonalization for cat pictures: anti-porn
    - so far only discussed how to define a metric to evaluate classifiers
    - separately about how to do well on the metric
    - distinct steps
- dev/test set could be different from user images
- if doing well on the metric and dev/test set does not correspond to doing well on the application, change the metric and/or dev/test set

Why Human-level Performance?

- workflow is much more efficient when compared to problem humans can do
- Comparing to human-level performance
    - approaches Bayes optimal error

    - slows down after human level performance
    - there are certain tools to use before human level performance vs after
- Why compare to human-level performance
    - humans are quite good at a lot of tasks
    - so long as ML is worse than humans
        - get labeled data from human
        - gain insight from manual error analysis: why did the person get this right
        - better analysis of bias/variance

Avoidable Bias

- if there is a huge gap between training error and human level performance
- alg not fitting training set well, bias → bigger NN, longer training ..
- if human performance is close to training error
- reduce the variance → regularization …
- avoidable bias is the minimum level of error between human level and training error

Understanding Human-level Performance

- Human-level error as a proxy for Bayes error
    - suppose a typical human achieves 3% error
    - typical doctor achieves 1% error
    - experienced doctor achieves 0.7%
    - team of experienced doctors achieves 0.5%
    - since human level performance is a proxy for Bayes error, the optimal is less than 0.5%, then 0.5% will be defined as human level error
    - be clear on what human-level error is
- Error analysis example
    - training error: 5%
    - dev error: 6%
    - human error: ~1%
    - ~4% avoidable bias
    - variance of 1%
    - so focus on training the network

Surpassing Human-level Performance

- what’s the avoidable bias if the training error is 0.3%, dev error is 0.4% and the human limit is 0.5%
- not enough information
- tools change
- Problems where ML significantly surpasses human-level performance
    - online advertising
    - product recommendations
    - logistics (travel time)
    - loan approvals
    - structural data
    - not natural perception problems
    - lots of data

Improving your Model Performance

- The two fundamental assumptions of supervised learning
    - you can fit the training set pretty well
    - the training set performance generalizes pretty well tot he dev/test set
- Reducing bias and variance
    - avoidable bias
        - train bigger model
        - train longer/better optimization algs
        - NN architecture/hyper search
    - variance
        - more data
        - regularization
            - L2, dropout, data augmentation
            - NN architecture/hyper search

Andrej Karpathy Interview

- has a stanford online class
- implementing from scratch