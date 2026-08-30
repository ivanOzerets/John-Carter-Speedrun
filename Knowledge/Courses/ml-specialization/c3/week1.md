What is clustering?

- Supervised learning
    - learn decision boundary
    - dataset included both inputs x and target outputs y
- Unsupervised learning
    - just given input labels x

    - clustering alg looks for groups
- Applications of clustering
    - grouping similar news
    - market segmentation
    - dna data
    - astronomical data analysis

K-means intuition

- takes a random guess of where the centers of the clusters may be → cluster centroids
- repeats two things
    - assign points to cluster centroids → go through each example and check which cluster centroid it is closer to
    - move cluster centroids → take average of position of on cluster centroid examples and move the centroid to the new position
- when there are no further changes, the points have converged

Optimization objective

- K-means alg
    - randomly initialize K cluster centroids mui_1, mui_2, …, mui_k
    - mui has the same dimension as training examples

    - eliminate a cluster if it has no examples in it or randomly reinitialize
- K-means for clusters that are not well separated
    - data may vary continuously

Initializing K-means

- K-means optimization objective
    - c^i = index of cluster to which example x^i is currently assigned
    - mui_k = cluster centroid k
    - mui_c(i) = cluster centroid of cluster to which example x^i has been assigned
    - example (x) → which index (c) → cluster location (mui)
    - cost function is the distance of the cluster centroid positions and the average of the positions in the cluster set

    - cost function J is called the distortion function
- Cost function for K-means
    - the first part of k-means updates c^1 - c^m while holding mui_1 - mui_K fixed
    - the second step is the reverse
- Moving the centroid
    - average squared distance minimizes distance from every example
    - if it goes up, there is a bug in the code
    - converged when the next iteration did not decrease

Initializing K-means

- Random initialization
    - choose K < m
    - randomly pick K training examples
    - set mui_1, …, mui_k equal to these K examples
- random initialization can run into local minima where the clusters aren’t right

- solution is to run multiple times → pick according to lowest cost junction J value
- Random initialization
    - run 50 - 1000 times is common

Choosing the number of clusters

- What is the right value of K?
    - ambiguous
- Choosing the value of K
    - elbow method
    - run K-means with many K
    - plot vs Cost function
    - as you increase the number of clusters → the cost function goes down
- the right “K” is often ambiguous
- K choice does not work to just choose what minimizes the cost function because then you would just choose the larger K

- Choosing the value of K
    - often, you want to get clusters for some later (downstream) purpose
    - evaluate K-means based on how well it performs on that later purpose
    - look at solutions vs tradeoffs

    - image compression
    - how good the image will look vs how large the size of the image is

Finding unusual events

- Anomaly detection
    - is there a anomaly with a new training example

- Density estimation
    - build model for probability of x being seen in the dataset

    - p(x_test) < epsilon → could be anomaly
- AD example
    - fraud detection
        - features of user i’s activities
    - manufacturing
        - airplane engine
        - circuit board
        - smartphone
    - features can be ratios → cpu load/ network traffic

Gaussian (normal/bell shaped) distribution

- Gaussian distribution
    - say x is a number
    - probability of x is determined by a Gaussian with mean mui, variance sigma^2 (variance)

- Example
    - when the distribution becomes skinnier, it also has to become taller

- Parameter estimation
    - Dataset: {x1, x2, …, xm}

    - maximum likelihood for mui, sigma

Anomaly detection alg

- Density estimation
    - each example x^i has n features
    - this alg works fine if the features are not statistically independent
    - probability of the model is the probability of each feature in terms of the features mean and variance

- Anomaly detection alg
    - Choose n features x_i that might be indicative of anomalous examples
    - Fit parameters mui_1, …, mui_n, sigma_1^2, …, sigma_n^2

    - Given new example x, compute P(x)

    - Anomaly if P(x) < epsilon
    - will flag a new example if one or more of the features of the examples is either very large or very small
- Anomaly detection example
    - creates gaussian distribution over the features space

Developing and evaluating an anomaly detection system

- The importance of real-number evaluation
    - when developing a learning alg (choosing features), making decisions is much easier if you have a way of evaluating the learning alg
    - assume we have some labeled data, of anomalous and non-anomalous examples
    - training set: (assume normal examples)
        - will do okay if a few split in
    - cross validation set: includes a few anomalous examples
    - test set: includes a few anomalous examples
- Aircraft engines monitoring example
    - 10000 good engines
    - 20 flawed engines
    - training set: 6000 good engines
    - CV: 2000 good engines, 10 anomalous
    - test: 2000 good engines, 10 anomalous
    - how many of anomalous engines the model correctly flags in the CV set
    - tune epsilon with CV set, tune features x_j
    - alternative: no test set
        - training set: 6000 good engines
        - CV: 4000 good engines (y = 0), 2 anomalous (y = 1), if not enough bad examples
        - higher risk of overfitting
- Algorithm evaluation
    - fit model p(x) on training set
    - on a cross validation/test example x, predict

    - use F1 score

Anomaly detection vs. supervised learning

- when you have a small number of positive examples, when should you use anomaly detection vs supervised learning
- Anomaly detection
    - very small number of positive examples (y = 1) (0-20 is common)
    - large number of negative (y = 0) examples
    - many different ‘types’ of anomalies, hard for any alg to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous examples so far
    - fraud
    - manufacturing: unseen defects
    - monitoring machines in a data center
- Supervised learning
    - Large number of positive and negative examples
    - enough positive examples for the alg to get a sense of what positive examples are like; future positive examples likely to be similar to ones in training set
    - spam email classification
    - manufacturing: seen defects (scratches)
    - weather prediction
    - diseases classification

Choosing what features to use

- Non-gaussian features
    - make sure the features are more or less gaussian, if not, transform
    - plt.hist(x) → plot examples

    - x _1= log(x_1)
    - x_2 = log(x_2 + 1)
    - x_3 = squ(x_3)
    - x_4 = x_4^(1/3)
    - apply same transformations to the CV and test set
- Error analysis for anomaly detection
    - want
        - p(x) ≥ epsilon large for normal examples x
        - p(x) < epsilon small for anomalous examples x
    - most common problem:
        - p(x) is comparable for normal and anomalous examples (p(x) large for both)
        - look at misclassified anomalies and determine new feature

- Monitoring computers in a data center
    - choose features that might take on unusually large or small values in the event of an anomaly
    - create new features by combining old features (high CPU load and low network traffic)