- evaluating a learning alg’s performance by measuring its training and cross validation error
- given these values, your are able to quantify how well a model is doing and this helps you make a decision on which one to use for a given application
- build upon this process and explore some tips to improve the performance of the models
- as it turns out, the training and cross validation errors can show what to try next to improve the models
- specifically, it will shown high bias or high variance

- the leftmost figure shows a high bias problem where the model is not capturing the patterns in the training data - high training and cross validation error
- the rightmost figure shows a high variance problem where the model has overfit the training set → low training error, but will perform poorly on new examples, indicated by a high cross validation error
- to fix a high bias problem
    - try adding polynomial features
    - try getting additional features
    - try decreasing the regularization parameter
- to fix a high variance problem
    - try increasing the regularization parameter
    - try smaller sets of features
    - get more training examples

Establishing Baseline Level of Performance

- before diagnosing a model for high bias or high variance, usually helpful to first have an idea of what level of error can be reasonably achieved
    - human level performance
    - copeting alg’s performance
    - guess based on experince
- often infeasible to get 0% error on real world noisy data

Imports and Lab Setup

- aside from a couple of linear regressors from scikit-learn, all other functions used in this lab are found in the utils.py
- mostly contains functions to split data, as well as function that loop over a list of parameters and plots the training and cross validation error for each one

    ```python
    # for building linear regression models
    from sklearn.linear_model import LinearRegression, Ridge

    # import lab utility functions in utils.py
    import utils 
    ```

Fixing High Bias

- Try adding polynomial features
    - adding polynomial features can help your model learn more complex patterns in the data
    - here is an example of a plot showing how the training and cross validation errors change as more polynomial features are added
    - use a synthetic dataset for a regression problem with one feature and one target
    - also define an arbitrary baseline performance and include it in the plot

        ```python
        # Split the dataset into train, cv, and test
        x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data1.csv')

        print(f"the shape of the training set (input) is: {x_train.shape}")
        print(f"the shape of the training set (target) is: {y_train.shape}\n")
        print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
        print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

        # Preview the first 5 rows
        print(f"first 5 rows of the training inputs (1 feature):\n {x_train[:5]}\n")

        # Instantiate the regression model class
        model = LinearRegression()

        # Train and plot polynomial regression models
        utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=400)

        '''
        the shape of the training set (input) is: (60, 1)
        the shape of the training set (target) is: (60,)

        the shape of the cross validation set (input) is: (20, 1)
        the shape of the cross validation set (target) is: (20,)

        first 5 rows of the training inputs (1 feature):
         [[3757.57575758]
         [2878.78787879]
         [3545.45454545]
         [1575.75757576]
         [1666.66666667]]
        '''
        ```

    - the more polynomial features, the better the model fits to the training data
    - the models with degree greater than four are low-bias because they perform close to or better than the baseline
    - if the baseline is defined lower, then the models are still considered high bias

        ```python
        # Train and plot polynomial regression models. Bias is defined lower.
        utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=250)
        ```

- Try getting additional features
    - try to acquire other features
    - another data collection campaign captures another feature

        ```python
        x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')

        print(f"the shape of the training set (input) is: {x_train.shape}")
        print(f"the shape of the training set (target) is: {y_train.shape}\n")
        print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
        print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

        # Preview the first 5 rows
        print(f"first 5 rows of the training inputs (2 features):\n {x_train[:5]}\n")

        '''
        the shape of the training set (input) is: (60, 2)
        the shape of the training set (target) is: (60,)

        the shape of the cross validation set (input) is: (20, 2)
        the shape of the cross validation set (target) is: (20,)

        first 5 rows of the training inputs (2 features):
         [[3.75757576e+03 5.49494949e+00]
         [2.87878788e+03 6.70707071e+00]
         [3.54545455e+03 3.71717172e+00]
         [1.57575758e+03 5.97979798e+00]
         [1.66666667e+03 1.61616162e+00]]
        '''
        ```

        ```python
        # Instantiate the model class
        model = LinearRegression()

        # Train and plot polynomial regression models. Dataset used has two features.
        utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=6, baseline=250)
        ```

    - the training error is now closer to the baseline
- Try decreasing the regularization parameter
    - introduce regularization to avoid overfitting
    - watch out for setting the regularization parameter too high
    - the cell below trains a 4th degree polynomial model using the Ridge class which allos you to set a regularization parameter

        ```python
        # Define lambdas to plot
        reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]

        # Define degree of polynomial and train for each value of lambda
        utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
        ```

    - the plot shows an initial lambda of 10 and the training error is worse than the baseline
    - this implies that it is placing a huge penalty on the w parameters and this prevents the model from learning more complex patterns in the data

Fixing High Variance

- the main objective is to have a model that generalizes well to new examples so you want to minimize the cross validation error
- Try increasing the regularization parameter
    - setting a very small value of the regularization parameter will keep the model low bias but might not do much to improve the variance
    - you can improve the cross validation error by increasing the value of lambda

        ```python
        # Define lambdas to plot
        reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

        # Define degree of polynomial and train for each value of lambda
        utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
        ```

- Try smaller sets of features
    - having too many polynomial terms can result in overfitting
    - reducing the number of such terms and see where you get the best balance of training and cross validation error
    - another scenario where reducing the number of features would be helpful is when you have irrelevant features in the data

        ```python
        # Prepare dataset with randomID feature
        x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')

        # Preview the first 5 rows
        print(f"first 5 rows of the training set with 2 features:\n {x_train[:5]}\n")

        # Prepare dataset with randomID feature
        x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data3.csv')

        # Preview the first 5 rows
        print(f"first 5 rows of the training set with 3 features (1st column is a random ID):\n {x_train[:5]}\n")

        '''
        first 5 rows of the training set with 2 features:
         [[3.75757576e+03 5.49494949e+00]
         [2.87878788e+03 6.70707071e+00]
         [3.54545455e+03 3.71717172e+00]
         [1.57575758e+03 5.97979798e+00]
         [1.66666667e+03 1.61616162e+00]]

        first 5 rows of the training set with 3 features (1st column is a random ID):
         [[1.41929130e+07 3.75757576e+03 5.49494949e+00]
         [1.51868310e+07 2.87878788e+03 6.70707071e+00]
         [1.92662630e+07 3.54545455e+03 3.71717172e+00]
         [1.25222490e+07 1.57575758e+03 5.97979798e+00]
         [1.76537960e+07 1.66666667e+03 1.61616162e+00]]
        '''
        ```

    - train the models and plot the results
    - the solid lines in the plot show the errors for the data with two features while the dotted lines show the error for the dataset with three
    - the one with three features has higher cross validation error especially as you introduce more polynomial terms
    - this is because the model is also trying to learn from the random IDs even though it has nothing to do with the target
    - another way to look at it is to observe the points at degree=4
    - notice that even though the training error is lower with three features, the gap between the training error and cross validation error is a lot wider when you only use 2 features

        ```python
        # Define the model
        model = LinearRegression()

        # Define properties of the 2 datasets
        file1 = {'filename':'data/c2w3_lab2_data3.csv', 'label': '3 features', 'linestyle': 'dotted'}
        file2 = {'filename':'data/c2w3_lab2_data2.csv', 'label': '2 features', 'linestyle': 'solid'}
        files = [file1, file2]

        # Train and plot for each dataset
        utils.train_plot_diff_datasets(model, files, max_degree=4, baseline=250)
        ```

- Get more training examples
    - try to minimize the cross validation error by getting more examples
    - train a 4th degree polynomial model then plot the learning curve of the model to see how the error behave with more examples

        ```python
        # Prepare the dataset
        x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data4.csv')
        print(f"the shape of the entire training set (input) is: {x_train.shape}")
        print(f"the shape of the entire training set (target) is: {y_train.shape}\n")
        print(f"the shape of the entire cross validation set (input) is: {x_cv.shape}")
        print(f"the shape of the entire cross validation set (target) is: {y_cv.shape}\n")

        # Instantiate the model class
        model = LinearRegression()

        # Define the degree of polynomial and train the model using subsets of the dataset.
        utils.train_plot_learning_curve(model, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)

        '''
        the shape of the entire training set (input) is: (600, 2)
        the shape of the entire training set (target) is: (600,)

        the shape of the entire cross validation set (input) is: (200, 2)
        the shape of the entire cross validation set (target) is: (200,)
        '''
        ```

        [https://www.notion.so](https://www.notion.so)

    - the cross validation error starts to approach the training error as the dataset size increases
    - another insight from this is that adding more examples will not likely solve a high bias problem
    - training error remains relatively flat even as the dataset increases