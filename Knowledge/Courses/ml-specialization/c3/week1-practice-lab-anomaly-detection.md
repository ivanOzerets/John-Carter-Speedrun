Packages

- import packages

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import *

    %matplotlib inline
    ```

Anomaly detection

- Problem Statement
    - implement an anomaly detection alg to detect anomalous behavior in server computers
    - the dataset contains two features
        - throughput
        - latency of response of each server
    - while the servers were operating, m = 307 examples are collected of how they were behaving, and thus have an unlabeled dataset {x^1, …, x^m}
        - the vast majority of these examples are ‘normal’ examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset
    - use a gaussian model to detect anomalous examples in the dataset
        - first start on a 2D dataset that will visualize what the alg is doing
        - on that dataset, fit a gaussian distribution and then find values that have very low probability and hence can be considered anomalies
        - apply the anomaly detection alg to a larger dataset with many dimensions
- Dataset
    - start by loading the dataset
        - the load_data() function shown below loads the data into the variables X_train, X_val and y_val
            - use X_train to fit a gaussian distribution
            - use X_val and y_val as a cross validation set to select a threshold and determine anomalous vs normal examples

            ```python
            # Load the dataset
            X_train, X_val, y_val = load_data()
            ```

    - View the Variables
        - print the first five elements of each of the variables

            ```python
            # Display the first five elements of X_train
            print("The first 5 elements of X_train are:\n", X_train[:5])  

            # Display the first five elements of X_val
            print("The first 5 elements of X_val are\n", X_val[:5])  

            # Display the first five elements of y_val
            print("The first 5 elements of y_val are\n", y_val[:5])  

            '''
            The first 5 elements of X_train are:
             [[13.04681517 14.74115241]
             [13.40852019 13.7632696 ]
             [14.19591481 15.85318113]
             [14.91470077 16.17425987]
             [13.57669961 14.04284944]]

             The first 5 elements of X_val are
             [[15.79025979 14.9210243 ]
             [13.63961877 15.32995521]
             [14.86589943 16.47386514]
             [13.58467605 13.98930611]
             [13.46404167 15.63533011]]

             The first 5 elements of y_val are
             [0 0 0 0 0]
            '''
            ```

    - Check the dimensions of the variables
        - print the shape of X_train, X_val and y_val

            ```python
            print ('The shape of X_train is:', X_train.shape)
            print ('The shape of X_val is:', X_val.shape)
            print ('The shape of y_val is: ', y_val.shape)

            '''
            The shape of X_train is: (307, 2)
            The shape of X_val is: (307, 2)
            The shape of y_val is:  (307,)
            '''
            ```

    - Visualize the data
        - use a scatter plot to visualize the data, since it has only two properties to plot (throughput and latency)

            ```python
            # Create a scatter plot of the data. To change the markers to blue "x",
            # we used the 'marker' and 'c' parameters
            plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

            # Set the title
            plt.title("The first dataset")
            # Set the y-axis label
            plt.ylabel('Throughput (mb/s)')
            # Set the x-axis label
            plt.xlabel('Latency (ms)')
            # Set axis range
            plt.axis([0, 30, 0, 30])
            plt.show()ni
            ```

    - Gaussian distribution
        - to perform anomaly detection, first fit a model to the data’s distribution
            - given a training set, estimate the Gaussian distribution for each of the features  x_i
            - recall that the Gaussian distribution is given by

                where mui is the mean and sigma^2 is the variance

            - each feature i = 1 … n, find parameters mui_i and sigma_i^2 that fit the data in the i-th dimension {x_i^1, …, x_i^m}
        - Estimating parameters for a Gaussian distribution
            - Implementation
                - complete the code in estimate_gaussian below
            - Exercise 1
                - complete the estimate_gaussian function below to calculate mu (mean for each feature in X) and var (variance for each feature in X)
                - you can estimate the parameters, of the i-th feature by using the following equations
                - to estimate the mean

                - for the variance

                    ```python
                    # UNQ_C1
                    # GRADED FUNCTION: estimate_gaussian

                    def estimate_gaussian(X): 
                        """
                        Calculates mean and variance of all features 
                        in the dataset

                        Args:
                            X (ndarray): (m, n) Data matrix

                        Returns:
                            mu (ndarray): (n,) Mean of all features
                            var (ndarray): (n,) Variance of all features
                        """

                        m, n = X.shape

                        ### START CODE HERE ### 
                        mu = np.mean(X, axis=0)
                        print(mu)
                        var = np.var(X, axis=0)
                        ### END CODE HERE ### 

                        return mu, var
                    ```

                    ```python
                    # Estimate mean and variance of each feature
                    mu, var = estimate_gaussian(X_train)              

                    print("Mean of each feature:", mu)
                    print("Variance of each feature:", var)

                    # UNIT TEST
                    from public_tests import *
                    estimate_gaussian_test(estimate_gaussian)

                    '''
                    [14.11222578 14.99771051]
                    Mean of each feature: [14.11222578 14.99771051]
                    Variance of each feature: [1.83263141 1.70974533]
                    [1. 2. 3.]
                    [2. 4. 6.]
                    [0.00773447 1.04476872 3.02069367]
                    All tests passed!
                    '''
                    ```

                - visualize the contours of the fitted gaussian distr.
                - most of the examples are in the region with the highest probability, while the anomalous examples are in the regions with lower prob.

                    ```python
                    # Returns the density of the multivariate normal
                    # at each data point (row) of X_train
                    p = multivariate_gaussian(X_train, mu, var)

                    #Plotting code 
                    visualize_fit(X_train, mu, var)
                    ```

            - Selecting the threshold eplison
                - investigate which examples have a very high probability given this distribution and which examples have a very low prob
                    - the low prob ex. are more likely to be the anomalies in the dataset
                    - one way to determine which examples are anomalies is to select a threshold based on a cross val set
                - complete the code in select_threshold to select the threshold epsilon using the F_1 score on a cross val set
                    - use a cross val. set where the label y = 1, corresponds to an anomalous example, and y = 0 corresponds to a normal ex
                    - for each cross val example, compute p(x_cv^i)
                    - the vector of all of these probabilities is passed to select_threshold in the vector p_val
                    - the corresponding labels are passed to the same function in the vector y_val
            - Exercise 2
                - complete the select_threshold function below to find the best threshold to use for selecting outliers based on the results from the val. set (p_val) and the ground truth (y_val)
                    - select_threshold has a loop that will try many different values of epsilon and select the best epsilon based on the F1 score
                    - implement code to calculate the F1 score from choosing epsilon as the threshold and place the value in F1
                        - if an example x has a low prob p(x) < epsilon, then it is classified as an anomaly
                        - compute precision and recall

                            - where tp is the number of true positives: the ground truth label says it’s an anomaly and the alg correctly classified it as an anomaly
                            - fp is the number of false positives: the ground truth label says it’s not an anomaly, but the alg incorrectly classified it as an anomaly
                            - fn is the number of false negatives: the ground truth label says it’s an anomaly, but the alg incorrectly classified it as not being anomalous
                        - The F1 score is computed using precision and recall as follows

                - Implementation Note: in order to compute tp, fp, and fn, use a vectorized implementation rather than loop over all the ex.

                    ```python
                    # UNQ_C2
                    # GRADED FUNCTION: select_threshold

                    def select_threshold(y_val, p_val): 
                        """
                        Finds the best threshold to use for selecting outliers 
                        based on the results from a validation set (p_val) 
                        and the ground truth (y_val)

                        Args:
                            y_val (ndarray): Ground truth on validation set
                            p_val (ndarray): Results on validation set

                        Returns:
                            epsilon (float): Threshold chosen 
                            F1 (float):      F1 score by choosing epsilon as threshold
                        """ 

                        best_epsilon = 0
                        best_F1 = 0
                        F1 = 0

                        step_size = (max(p_val) - min(p_val)) / 1000

                        for epsilon in np.arange(min(p_val), max(p_val), step_size):

                            ### START CODE HERE ###

                            preds = (p_val < epsilon)

                            # binary and because two binary vectors
                            tp = sum((preds == 1) & (y_val == 1)) 
                            fp = sum((preds == 1) & (y_val == 0))
                            fn = sum((preds == 0) & (y_val == 1))

                            prec = tp / (tp + fp)
                            rec = tp / (tp + fn)

                            F1 = (2 * prec * rec) / (prec + rec)

                            ### END CODE HERE ### 

                            if F1 > best_F1:
                                best_F1 = F1
                                best_epsilon = epsilon

                        return best_epsilon, best_F1
                    ```

                    ```python
                    p_val = multivariate_gaussian(X_val, mu, var)
                    epsilon, F1 = select_threshold(y_val, p_val)

                    print('Best epsilon found using cross-validation: %e' % epsilon)
                    print('Best F1 on Cross Validation Set: %f' % F1)

                    # UNIT TEST
                    select_threshold_test(select_threshold)

                    '''
                    Best epsilon found using cross-validation: 8.990853e-05
                    Best F1 on Cross Validation Set: 0.875000
                    All tests passed!
                    '''
                    ```

                - run the anomaly detection code and circle the anomalies

                    ```python
                    # Find the outliers in the training set 
                    outliers = p < epsilon

                    # Visualize the fit
                    visualize_fit(X_train, mu, var)

                    # Draw a red circle around those outliers
                    plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
                             markersize= 10,markerfacecolor='none', markeredgewidth=2)
                    ```

    - High dimensional dataset
        - run the anomaly detection alg on a more realistic and much harder dataset
        - each example is described by 11 features, capturing many more properties of the compute servers
        - load the dataset
            - load_data() function loads the data into variables X_train_high, X_val_high, andy_val_high
                - _high is meant to distinguish from the ones previous
                - use X_train_high to fit Gaussian dist.
                - use X_val_high and y_val_high as a cross val set to select the threshold and determine anomalous vs normal examples

                ```python
                # load the dataset
                X_train_high, X_val_high, y_val_high = load_data_multi()
                ```

        - Check the dimensions of the variables
            - dimensions of the new variables

                ```python
                print ('The shape of X_train_high is:', X_train_high.shape)
                print ('The shape of X_val_high is:', X_val_high.shape)
                print ('The shape of y_val_high is: ', y_val_high.shape)

                '''
                The shape of X_train_high is: (1000, 11)
                The shape of X_val_high is: (100, 11)
                The shape of y_val_high is:  (100,)
                '''
                ```

        - Anomaly detection
            - run the anomaly detection on the new dataset
                - estimate the gaussian parameters
                - evaluate the probs for both the training data X_train_high from which were estimated the gaussian params, as well as for the cross-val set X-val_high
                - finally, it will use select_threshold to find the best threshold epsilon

                ```python
                # Apply the same steps to the larger dataset

                # Estimate the Gaussian parameters
                mu_high, var_high = estimate_gaussian(X_train_high)

                # Evaluate the probabilites for the training set
                p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

                # Evaluate the probabilites for the cross validation set
                p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

                # Find the best threshold
                epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

                print('Best epsilon found using cross-validation: %e'% epsilon_high)
                print('Best F1 on Cross Validation Set:  %f'% F1_high)
                print('# Anomalies found: %d'% sum(p_high < epsilon_high))

                '''
                [  4.93940034  -9.63726819  13.81470749 -10.4644888   -7.95622922
                  10.19950372  -6.01940755   7.96982896  -6.2531819    2.32451289
                   8.47372252]
                Best epsilon found using cross-validation: 1.377229e-18
                Best F1 on Cross Validation Set:  0.615385
                # Anomalies found: 117
                '''
                ```