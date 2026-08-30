- visualize how a decision tree is split using information gain
- we decide if a node will be split or not by looking at the information gain that split would give us
- where information gain is

- and h is the entropy

- remember that log is defined to be in base 2
- note that the H attains its higher value when p = 0.5
- this means that the probability of even tis 0.5
- its minimum value is attained in p=0 and p =1
- the probability of the event happening is totally predictable

- use one-hot encoding to encode the categorical features
    - ear shape: pointy = 1, floppy = 0
    - face shape: round = 1, not round = 0
    - whiskers: present = 1, absent = 0
- two sets:
    - x_train: for each example, contains 3 features
        - ear shape, 1 point, 0 no
        - face shape, 1 round, 0 no
        - whiskers, 1 present, 0 no
    - y_train: whether the animal is a cat
        - 1 is yes
        - 0 no

    ```python
    X_train = np.array([[1, 1, 1],
    [0, 0, 1],
     [0, 1, 0],
     [1, 0, 1],
     [1, 1, 1],
     [1, 1, 0],
     [0, 0, 0],
     [1, 1, 0],
     [0, 1, 0],
     [0, 1, 0]])

    y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    ```

    ```python
    #For instance, the first example
    X_train[0]

    #array([1, 1, 1])
    ```

- on each node, compute the information gain for each feature, then split the node on the feature with the higher information gain, by comparing the entropy of the node with the weighted entropy in the two splitted nodes
- the root node has every animal in the dataset
- p_1^node is the proportion of positive class in the root node

- write a function to compute the entropy

    ```python
    def entropy(p):
        if p == 0 or p == 1:
            return 0
        else:
            return -p * np.log2(p) - (1- p)*np.log2(1 - p)

    print(entropy(0.5))

    # 1.0
    ```

- compute the information gain for each of the features

    ```python
    def split_indices(X, index_feature):
        """Given a dataset and a index feature, return two lists for
        the two split nodes, the left node has the animals that have 
        that feature = 1 and the right node those that have the feature = 0 
        index feature = 0 => ear shape
        index feature = 1 => face shape
        index feature = 2 => whiskers
        """
        left_indices = []
        right_indices = []
        for i,x in enumerate(X):
            if x[index_feature] == 1:
                left_indices.append(i)
            else:
                right_indices.append(i)
        return left_indices, right_indices
    ```

- choosing ear shape to split

    ```python
    split_indices(X_train, 0)

    # ([0, 3, 4, 5, 7], [1, 2, 6, 8, 9])
    ```

- another function to compute the weighted entropy in the splitted noes
- need to find
    - w_left and w_right, the proportion of animals in each node
    - p_left and p_right, the proportion of cats in each split
- note the difference between these two definitions
- if the root node is split on the feature of index 0, then in the left node, the one that has the animals 0, 3, 4, 5, and 7

    ```python
    def weighted_entropy(X,y,left_indices,right_indices):
        """
        This function takes the splitted dataset, the indices we chose to split and
        returns the weighted entropy.
        """
        w_left = len(left_indices)/len(X)
        w_right = len(right_indices)/len(X)
        p_left = sum(y[left_indices])/len(left_indices)
        p_right = sum(y[right_indices])/len(right_indices)

        weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
        return weighted_entropy
    ```

    ```python
    left_indices, right_indices = split_indices(X_train, 0)
    weighted_entropy(X_train, y_train, left_indices, right_indices)

    # 0.7219280948873623
    ```

- to compute the information gain, subtract it from the entropy in the node chosen to split

    ```python
    def information_gain(X, y, left_indices, right_indices):
        """
        Here, X has the elements in the node and y is theirs respectives classes
        """
        p_node = sum(y)/len(y)
        h_node = entropy(p_node)
        w_entropy = weighted_entropy(X,y,left_indices,right_indices)
        return h_node - w_entropy
    ```

    ```python
    information_gain(X_train, y_train, left_indices, right_indices)

    #0.2780719051126377
    ```

- compute the information gain for each feature split on the root node

    ```python
    for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
        left_indices, right_indices = split_indices(X_train, i)
        i_gain = information_gain(X_train, y_train, left_indices, right_indices)
        print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")

    '''
    Feature: Ear Shape, information gain if we split the root node using this feature: 0.28
    Feature: Face Shape, information gain if we split the root node using this feature: 0.03
    Feature: Whiskers, information gain if we split the root node using this feature: 0.12
    '''
    ```

- the best feature to split is indeed the ear shape
- the process is recursive, which means these calculations are performed for each node until a stopping criteria is met
    - if the tree depth after splitting exceeds a threshold
    - if the resulting node has only 1 class
    - if the information gain of splitting is below a threshold