Packages

- numpy
- matplotlib
- util.py

```python
import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

%matplotlib inline
```

Problem Statement

- suppose you are starting a company that grows and sells wild mushrooms
    - since not all mushrooms are edible, you’d like to be able to tell whether a given mushroom is edible or poisonous based on it's physical attributes
    - you have some existing data that you can use for this task
    - can you use the data to help you identify which mushrooms can be sold safely?

Dataset

- load the dataset

- You have 10 examples of mushrooms. For each example, you have
    - Three features
        - Cap Color (brown or red),
        - Stalk Shape (Tapering (as in \/) or Enlarging (as in /\), and
        - Solitary (Yes or No)
    - Label
        - edible (1 indicating yes or 0 indicating poisonous)
- One-hot encoded dataset
    - for ease of implemntation, one-hot encoded the feature

    - x_train contains three features for each example
        - Brown Color (A value of 1 indicates "Brown" cap color and 0 indicates "Red" cap color)
        - Tapering Shape (A value of 1 indicates "Tapering Stalk Shape" and 0 indicates "Enlarging" stalk shape)
        - Solitary (A value of 1 indicates "Yes" and 0 indicates "No")
    - y_train is whether the mushroom is edible
        - y=1 indicates edible
        - y=0 indicates poisonous

        ```python
        X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
        y_train = np.array([1,1,0,0,1,0,0,1,1,0])
        ```

    - View the variables
        - print the first few elements of X_train and the type of the variable

            ```python
            print("First few elements of X_train:\n", X_train[:5])
            print("Type of X_train:",type(X_train))

            '''
            First few elements of X_train:
             [[1 1 1]
             [1 0 1]
             [1 0 0]
             [1 0 0]
             [1 1 1]]
            Type of X_train: <class 'numpy.ndarray'>
            '''
            ```

        - same for y_train

            ```python
            print("First few elements of y_train:", y_train[:5])
            print("Type of y_train:",type(y_train))

            '''
            First few elements of y_train: [1 1 0 0 1]
            Type of y_train: <class 'numpy.ndarray'>
            '''
            ```

    - Check the dimensions of the variables
        - view its dimensions
        - print the shape of X_train and y_train and see how many training examples are in the dataset

            ```python
            print ('The shape of X_train is:', X_train.shape)
            print ('The shape of y_train is: ', y_train.shape)
            print ('Number of training examples (m):', len(X_train))

            '''
            The shape of X_train is: (10, 3)
            The shape of y_train is:  (10,)
            Number of training examples (m): 10
            '''
            ```

- Decision Tree Refresher
    - recall that the steps for building a decision tree are as follows:
        - start with all examples at the root node
        - calculate information gain for splitting on all possible features, and pick the one with the highest information gain
        - split dataset according to the selected feature, and create left and right branches of the tree
        - keep repeating splitting process until stopping criteria is met
    - in this lab, you'll implement the following functions, which will let you split a node into left and right branches using the feature with the highest information gain
        - calculate the entropy at a node
        - split the dataset at a node into left and right branches based on a given feature
        - calculate the information gain from splitting on a given feature
        - choose the feature that maximizes information gain
    - we'll then use the helper functions you've implemented to build a decision tree by repeating the splitting process until the stopping criteria is met
        - for this lab, the stopping criteria we've chosen is setting a maximum depth of 2
    - Calculate entropy
        - write a helper function called compute_entropy that computes the entropy (measure of impurity) at a node
            - the function takes in a numpy array (y) that indicates whether the examples in that node are edible (1) or poisonous(0)
        - complete the compute_entropy() function below to
            - compute p1, which is the fraction of examples that are edible
            - the entropy is then calculated as

            - note
                - the log is calculated with base 2
                - for implementation purposed, 0log_2(0) = 0
                - make sure to check that the data at a node is not empty
                - return 0 if it is
    - Exercise 1
        - complete the compute_entropy() function

        ```python
        # UNQ_C1
        # GRADED FUNCTION: compute_entropy

        def compute_entropy(y):
            """
            Computes the entropy for 

            Args:
               y (ndarray): Numpy array indicating whether each example at a node is
                   edible (`1`) or poisonous (`0`)

            Returns:
                entropy (float): Entropy at that node

            """
            # You need to return the following variables correctly
            entropy = 0.

            ### START CODE HERE ###

            if len(y) != 0:
                p1 = len(y[y==1])/len(y)

                if p1 != 0 and p1 != 1:
                    entropy = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)

            ### END CODE HERE ###        

            return entropy
        ```

        ```python
        # Compute entropy at the root node (i.e. with all examples)
        # Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"

        print("Entropy at root node: ", compute_entropy(y_train)) 

        # UNIT TESTS
        compute_entropy_test(compute_entropy)

        '''
        Entropy at root node:  1.0
         All tests passed. 
        '''
        ```

    - Split dataset
        - write a helper function called split_dataset that takes in the data at a node and a feature to split on and splits it into left and right branches
        - the function takes in the training data, the list of indices of data points at that node, along with the feature to split on
        - it splits the data and returns the subset of indices at the left and the right branch.
        - for example, say we're starting at the root node (so node_indices = [0,1,2,3,4,5,6,7,8]), and we chose to split on feature 00, which is whether or not the example has a brown cap
            - the output of the function is then, left_indices = [0,1,2,3,4,7,8,9] (data points with brown cap) and right_indices = [5,6,8] (data points without a brown cap)
- Exercise 2
    - complete the split_dataset() function
    - For each index in node_indices
        - If the value of X at that index for that feature is 1, add the index to left_indices
        - If the value of X at that index for that feature is 0, add the index to right_indices

        ```python
        # UNQ_C2
        # GRADED FUNCTION: split_dataset

        def split_dataset(X, node_indices, feature):
            """
            Splits the data at the given node into
            left and right branches

            Args:
                X (ndarray):             Data matrix of shape(n_samples, n_features)
                node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
                feature (int):           Index of feature to split on

            Returns:
                left_indices (list):     Indices with feature value == 1
                right_indices (list):    Indices with feature value == 0
            """

            # You need to return the following variables correctly
            left_indices = []
            right_indices = []

            ### START CODE HERE ###
            for i in node_indices:
                if X[i][feature]:
                    left_indices.append(i)
                else:
                    right_indices.append(i)

            ### END CODE HERE ###

            return left_indices, right_indices
        ```

        ```python
        # Case 1

        root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Feel free to play around with these variables
        # The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
        feature = 0

        left_indices, right_indices = split_dataset(X_train, root_indices, feature)

        print("CASE 1:")
        print("Left indices: ", left_indices)
        print("Right indices: ", right_indices)

        # Visualize the split 
        generate_split_viz(root_indices, left_indices, right_indices, feature)

        print()

        # Case 2

        root_indices_subset = [0, 2, 4, 6, 8]
        left_indices, right_indices = split_dataset(X_train, root_indices_subset, feature)

        print("CASE 2:")
        print("Left indices: ", left_indices)
        print("Right indices: ", right_indices)

        # Visualize the split 
        generate_split_viz(root_indices_subset, left_indices, right_indices, feature)

        # UNIT TESTS    
        split_dataset_test(split_dataset)

        '''
        CASE 1:
        Left indices:  [0, 1, 2, 3, 4, 7, 9]
        Right indices:  [5, 6, 8]
        CASE 2:
        Left indices:  [0, 2, 4]
        Right indices:  [6, 8]
        All tests passed.
        '''
        ```

    - Calculate information gain
        - write a function called information_gain that takes in the training data, the indices at a node and a feature to split on and return the information gain from the split
- Exercise 3
    - complete the compute_information_gain() function where

    - H(p1^node) is entropy at the node
    - H(p1^left) and H(p1^right) are the entropies at the left and the right branches resulting from the split
    - w^left and w^right are the proportion of examples at the left and right branch
    - Note:
        - You can use the compute_entropy() function that you implemented above to calculate the entropy
        - We've provided some starter code that uses the split_dataset() function you implemented above to split the dataset

    ```python
    # UNQ_C3
    # GRADED FUNCTION: compute_information_gain

    def compute_information_gain(X, y, node_indices, feature):

        """
        Compute the information of splitting the node on a given feature

        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            feature (int):           Index of feature to split on

        Returns:
            cost (float):        Cost computed

        """    
        # Split dataset
        left_indices, right_indices = split_dataset(X, node_indices, feature)

        # Some useful variables
        X_node, y_node = X[node_indices], y[node_indices]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # You need to return the following variables correctly
        information_gain = 0

        ### START CODE HERE ###
        hp1n = compute_entropy(y_node)
        hp1l = compute_entropy(y_left)
        hp1r = compute_entropy(y_right)

        wleft = len(X_left)/len(X_node)
        wright = len(X_right)/len(X_node)

        information_gain = hp1n - (wleft * hp1l + wright * hp1r)

        ### END CODE HERE ###  

        return information_gain
    ```

    ```python
    info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
    print("Information Gain from splitting the root on brown cap: ", info_gain0)

    info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
    print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

    info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
    print("Information Gain from splitting the root on solitary: ", info_gain2)

    # UNIT TESTS
    compute_information_gain_test(compute_information_gain)

    '''
    Information Gain from splitting the root on brown cap:  0.034851554559677034
    Information Gain from splitting the root on tapering stalk shape:  0.12451124978365313
    Information Gain from splitting the root on solitary:  0.2780719051126377
     All tests passed.
    '''
    ```

    - Get best split
        - write a function to get the best feature to split on by computing the information gain from each feature as above and return the feature that gives the max information gain
- Exercise 4
    - compute the get_best_split() function
    - the function takes in the training data, along with the indices of datapoint at that node
    - the output of the function is the feature that gives the maximum information gain
        - you can use the compute_information_gain() function to iterate through the features and calculate the information for each feature

    ```python
    # UNQ_C4
    # GRADED FUNCTION: get_best_split

    def get_best_split(X, y, node_indices):   
        """
        Returns the optimal feature and threshold value
        to split the node data 

        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

        Returns:
            best_feature (int):     The index of the best feature to split
        """    

        # Some useful variables
        num_features = X.shape[1]

        # You need to return the following variables correctly
        best_feature = -1

        ### START CODE HERE ###

        max_info_gain = 0
        for feature in range(num_features):
            info_gain = compute_information_gain(X, y, node_indices, feature)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature

        ### END CODE HERE ##    

        return best_feature
    ```

    ```python
    best_feature = get_best_split(X_train, y_train, root_indices)
    print("Best feature to split on: %d" % best_feature)

    # UNIT TESTS
    get_best_split_test(get_best_split)

    '''
    Best feature to split on: 2
     All tests passed.
    '''
    ```

- Building the tree
    - use the function implemented above to generate a decision tree by successively picking the best feature to split on until reaching the stopping criteria (max depth is 2)

    ```python
    # Not graded
    tree = []

    def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
        """
        Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
        This function just prints the tree.

        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
            max_depth (int):        Max depth of the resulting tree. 
            current_depth (int):    Current depth. Parameter used during recursive call.

        """ 

        # Maximum depth reached - stop splitting
        if current_depth == max_depth:
            formatting = " "*current_depth + "-"*current_depth
            print(formatting, "%s leaf node with indices" % branch_name, node_indices)
            return

        # Otherwise, get best split and split the data
        # Get the best feature and threshold at this node
        best_feature = get_best_split(X, y, node_indices) 

        formatting = "-"*current_depth
        print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

        # Split the dataset at the best feature
        left_indices, right_indices = split_dataset(X, node_indices, best_feature)
        tree.append((left_indices, right_indices, best_feature))

        # continue splitting the left and the right child. Increment current depth
        build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
        build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
    ```

    ```python
    build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
    generate_tree_viz(root_indices, y_train, tree)

    '''
     Depth 0, Root: Split on feature: 2
    - Depth 1, Left: Split on feature: 0
      -- Left leaf node with indices [0, 1, 4, 7]
      -- Right leaf node with indices [5]
    - Depth 1, Right: Split on feature: 1
      -- Left leaf node with indices [8]
      -- Right leaf node with indices [2, 3, 6, 9]
    '''
    ```