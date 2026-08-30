Decision tree model

- Cat classification example
    - five cat and five dog dataset
    - features take on categorical values, discrete
    - binary classification task

- Decision Tree
    - model with nodes
    - decision nodes look at particular features deciding whether you go left or right down the tree
    - left notes and the predictions

    - goal is to pick the best decision tree that will perform the best

Learning Process

- Decision Tree Learning
    - decide what feature to use at the root node

- decision 1: how to choose what feature to split on at each node?
    - maximize purity (or minimize impurity)
    - cat DNA, for example, would be a great feature to get a pure split
- decision 2: when do you stop splitting?
    - when a node is 100% one class
    - when splitting a node will result in the tree exceeding a max depth
    - when improvements in purity score are below a threshold → to keep tree smaller and reduce risk of overfitting
    - when number of examples in a node is below a threshold

Measuring purity

- entropy → measure of impurity of a set of data
- p1 = fraction of examples that are cats
- entropy function H(p1)

- entropy is highest when the data set equal between positive and negative examples
- entropy is lowest when the data set is only one category
- p0 = 1 - p1

- the log is set to 2 to make the peak equal to 1
- Note “0log(0)” = 0
- other functions like genie criteria

Choosing a split: Information Gain

- what choice of feature reduces entropy the most
- the reduction of entropy is called information gain
- Choosing a split
    - splitting on a feature → compute entropy of p1
    - take a weighted average
    - worse if a lot of example with high entropy rather than small number of example with high entropy
    - combine sides into one number
    - compute weighted average for all splits and pick the lowest
    - convention in decision tree building is to compute reduction in entropy

    - resulting numbers are the information gain
    - one of the stopping criteria is if the reduction in entropy is too small or not enough information gain
- Information Gain
    - w_left and w_right is the fraction of examples that went to each branch respectively

Putting it together

- Decision Tree Learning
    - start with all examples at the root node
    - calculate information gain for all possible features, and pick the one with the highest information gain
    - split dataset according to selected feature, and create left and right branches of the tree
    - keep repeating splitting process until stopping criteria is met:
        - when a node is 100% one class
        - when splitting a node will result in the tree exceeding a maximum depth
        - information gain from additional splits is less than threshold
        - when number of examples in a node is below a threshold
- Recursive splitting
    - a recursive alg

    - the larger the max depth the bigger the decision tree → fitting higher degree polynomial like

Using one-hot encoding of categorical features

- what happens if you have features that take on more than two discrete values
- Features with three possible values
    - three sub branches but better to
    - split into three separate features

- One hot encoding
    - if a categorical feature can take on k values, create k binary features (0 or 1 valued)
    - exactly one of the features is 1

Continuous valued features

- Continuous features
    - weight for example
- Splitting on a continuous variable
    - sort examples according to value of features and take all values that are midpoint of sorted values under consideration

Regression Trees

- can predict numbers
- Regression with Decision Trees: Predicting a number
    - weight will be the target, for example
- Regression with Decision Trees
    - average example weights of cats at leaf nodes
- Choosing a split
    - instead of reducing entropy, reduce the variance of the weights of the values y
    - variance computes how widely a set of numbers vary
    - convention is to choose with the reduction in variance

    - train a lot of decision trees is called an ensemble

Using multiple decision trees

- Trees are highly sensitive to small changes of the data
    - on low data trees changing only one example changes the whole tree → alg is not robust
    - often get better results if you train a bunch of decision trees
- Tree ensemble
    - all trees predict new examples and vote

Sampling with replacement

- tokens get picked randomly and are put back into the ‘bag’
- the examples are put in a ‘bag’
- create a new random training set this way
- can have repeated examples

Random forest algorithm

- Generating a tree sample
    - given training set of size m
    - for b = 1 to B:
        - use sampling with replacement to create a new training set of size m
        - train a decision tree on the new dataset
    - B is anywhere from 64 - 128
    - bagged decision tree
    - not uncommon for all root nodes to be split with the same feature
- Randomizing the feature choice
    - at each node, when choosing a feature to use to split, if n features are available, pick a random subset of k < n features and allow the algorithm to only choose from that subset of features
    - k = sqrt(n)
    - random forest algorithm

XGBoost

- by far the most commonly used decision tree alg
- Boosted trees intuition
    - given training set of size m
    - for b = 1 to B:
        - Use sampling with replacement to create a new training set of size m
        - but instead of picking from all examples with equal (1/m) probability, make it more likely to pick misclassified examples from previously trained trees
        - train a decision tree on the new dataset
        - on each iteration look at what the ensemble of trees are not yet doing that well
- XGBoost (eXtreme Gradient Boosting)
    - open source implementation of boosted trees
    - fast efficient implementation
    - good choice of default splitting criteria and criteria for when to stop splitting
    - built in regularization to prevent overfitting
    - highly competitive alg for machine learning competitions (Kaggle)
- Using XGBoost
    - Classification

    - Regression

When to use decision trees

- Decision Trees vs NN
    - Decision Trees and Tree ensembles
        - works well on tabular (structured) data
        - not recommended for unstructured data (images, audio, text)
        - fast to train
        - small decision trees may be human interpretable
    - NN
        - works well on all types of data, including tabular (structured) and unstructured data
        - may be slower than a decision tree
        - works with transfer learning
        - when building a system of multiple models working together, it might be easier to string together multiple NN → you can train them all together rather than one at a time as in for decision trees