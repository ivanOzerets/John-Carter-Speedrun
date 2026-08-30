- Quantifying a learning alg’s performance and comparing different models are some of the common tasks when applying machine learning to real work applications

Imports and Lab Setup

- import the packages needed for the tasks
- included some commands to make the outputs later more readable by reducing verbosity and suppressing non-critical warnings

    ```python
    # for array computations and loading data
    import numpy as np

    # for building linear regression models and preparing data
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # for building and training neural networks
    import tensorflow as tf

    # custom functions
    import utils

    # reduce display precision on numpy arrays
    np.set_printoptions(precision=2)

    # suppress warnings
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    ```

Regression

- tasked to develop a model for a regression problem
- given the dataset below consisting of 50 examples of an input feature x and its corresponding target y

    ```python
    # Load the dataset from the text file
    data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')

    # Split the inputs and outputs into separate arrays
    x = data[:,0]
    y = data[:,1]

    # Convert 1-D arrays into 2-D because the commands later will require it
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    print(f"the shape of the inputs x is: {x.shape}")
    print(f"the shape of the targets y is: {y.shape}")

    '''
    the shape of the inputs x is: (50, 1)
    the shape of the targets y is: (50, 1)
    '''
    ```

- plot the dataset to get an idea of how the target behaves w.r.t the input

Split the dataset into training, cross validation, and test sets

- it is best to hold out a portion of the data to measure how well the model generalizes to new examples
    - training set - used to train the model
    - cross validation set (validation, development, or dev set) - used to evaluate the different model configurations, what polynomial features to add for example
    - test set - used to give a fair estimate of the chosen model’s performance against new examples, should not be used to make decisions while still developing the models
- Scikit-learn provides a train_test_split function to split the data into the parts mentioned

    ```python
    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

    # Delete temporary variables
    del x_, y_

    print(f"the shape of the training set (input) is: {x_train.shape}")
    print(f"the shape of the training set (target) is: {y_train.shape}\n")
    print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
    print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
    print(f"the shape of the test set (input) is: {x_test.shape}")
    print(f"the shape of the test set (target) is: {y_test.shape}")

    '''
    the shape of the training set (input) is: (30, 1)
    the shape of the training set (target) is: (30, 1)

    the shape of the cross validation set (input) is: (10, 1)
    the shape of the cross validation set (target) is: (10, 1)

    the shape of the test set (input) is: (10, 1)
    the shape of the test set (target) is: (10, 1)
    '''
    ```

- plot the dataset again below to see which points were used as training, cross validation, or test data

    ```python
    utils.plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="input vs. target")
    ```

Fit a linear model

- now that the data is split one of the first things to try is fit a linear model
- Feature scaling
    - usually a good idea to perform feature scaling to help the model converge faster
    - especially true if the input features have widely different ranges of values
    - adding polynomial terms will make the input features indeed have different ranges
    - for example, x runs from around 1600 to 3600, while x^2 will run from 2.56 million to 12.96 million
    - only use x for the first model but it’s good to practice feature scaling to apply later
    - use the StandardScalar class from scitkit-learn
    - this computes the z-score of the inputs, where mu is the mean of the feature values and sigma is the standard deviation

    - the code below shows how to prepare the training set
    - plot the results to inspect if it still follows the same pattern
    - the new graph should have a reduced range of values for x

        ```python
        # Initialize the class
        scaler_linear = StandardScaler()

        # Compute the mean and standard deviation of the training set then transform it
        X_train_scaled = scaler_linear.fit_transform(x_train)

        print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
        print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

        # Plot the results
        utils.plot_dataset(x=X_train_scaled, y=y_train, title="scaled input vs. target")

        '''
        Computed mean of the training set: 2504.06
        Computed standard deviation of the training set: 574.85
        '''
        ```

Train the model

- create and train a regression model
- use the LinearRegression class, but take note that there are other linear regressors which can also be used

```python
# Initialize the class
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_scaled, y_train )

# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
```

Evaluate the Model

- to evaluate the performance of your model, you will measure the error for the training and cross validation sets
- for the training error, recall the equation for calculating the mean squared error (MSE)

- scikit-learn also has a built-in mean_squared_error() function
- note that as per the documentation, scikit’learn’s implementation only divides by m and not 2m, where m is the number of examples
- dividing by 2m is a convention, but the calculations should still work whether or not it is included
- to match, just divide by 2
- note: since the model is trained on scaled values (using the z-score), also feed in the scaled training set instead of its raw values

    ```python
    # Feed the scaled training set and get the predictions
    yhat = linear_model.predict(X_train_scaled)

    # Use scikit-learn's utility function and divide by 2
    print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

    # for-loop implementation
    total_squared_error = 0

    for i in range(len(yhat)):
        squared_error_i  = (yhat[i] - y_train[i])**2
        total_squared_error += squared_error_i                                              

    mse = total_squared_error / (2*len(yhat))

    print(f"training MSE (for-loop implementation): {mse.squeeze()}")

    '''
    training MSE (using sklearn function): 406.19374192533155
    training MSE (for-loop implementation): 406.19374192533155
    '''
    ```

- you can then compute the MSE for the cross validation set with basically the same equation

- as with the training set, scale the cross validation set
- an important thing to note when using the z-score is you have to use the mean and standard deviation of the training set when scaling the cross validation set
- this is to ensure that the input features are transformed as expected by the model
- one way to gain intuition is with this scenario:
    - say that the training set has an input feature equal to 500 which is scaled down to 0.5 using the z-score
    - after training, your model is able to accurately map this scaled input x=0.5 to the target output y=300
    - now lets say a sample was fed to the model equal to 500
    - if you get this input sample’s z-score using any other values of the mean and standard deviation, then it might not be scaled to 0.5 and the model will most likely make a wrong prediction
- scale the cross validation set by using the same StandardScaler as earlier but only calling its transform() method instead of fit_transform()

    ```python
    # Scale the cross validation set using the mean and standard deviation of the training set
    X_cv_scaled = scaler_linear.transform(x_cv)

    print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
    print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")

    # Feed the scaled cross validation set
    yhat = linear_model.predict(X_cv_scaled)

    # Use scikit-learn's utility function and divide by 2
    print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")

    '''
    Mean used to scale the CV set: 2504.06
    Standard deviation used to scale the CV set: 574.85
    Cross validation MSE: 551.7789026952216
    '''
    ```

Adding Polynomial Features

- notice that the target y rises more sharply at smaller values of x compared to higher ones
- a straight line might not be the best choice because the target y seems to flatten out as x increases
- with the values of the training and cross validation MSE from the linear model, try adding polynomial features to try to get a better performance
- the code will mostly be the same but with a few extra preprocessing steps
- Create the additional features
    - first, generate the polynomial features from the training set
    - the code below demonstrates how to do this using the PolynomialFeatures class
    - it will create a new input feature which has the squared values of the inputs (degree=2)

        ```python
        # Instantiate the class to make polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)

        # Compute the number of features and transform the training set
        X_train_mapped = poly.fit_transform(x_train)

        # Preview the first 5 elements of the new training set. Left column is `x` and right column is `x^2`
        # Note: The `e+<number>` in the output denotes how many places the decimal point should 
        # be moved. For example, `3.24e+03` is equal to `3240`
        print(X_train_mapped[:5])

        '''
        [[3.32e+03 1.11e+07]
         [2.34e+03 5.50e+06]
         [3.49e+03 1.22e+07]
         [2.63e+03 6.92e+06]
         [2.59e+03 6.71e+06]]
        '''
        ```

    - then scale the inputs as before to narrow down the range of values

        ```python
        # Instantiate the class
        scaler_poly = StandardScaler()

        # Compute the mean and standard deviation of the training set then transform it
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

        # Preview the first 5 elements of the scaled training set.
        print(X_train_mapped_scaled[:5])

        '''
        [[ 1.43  1.47]
         [-0.28 -0.36]
         [ 1.71  1.84]
         [ 0.22  0.11]
         [ 0.15  0.04]]
        '''
        ```

    - then proceed to train the model
    - measure the model’s performance against the cross validation set
    - make sure to perform the same transformations done to the training set
    - add the same number of polynomial features, then scale the range of values

        ```python
        # Initialize the class
        model = LinearRegression()

        # Train the model
        model.fit(X_train_mapped_scaled, y_train )

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2}")

        # Add the polynomial features to the cross validation set
        X_cv_mapped = poly.transform(x_cv)

        # Scale the cross validation set using the mean and standard deviation of the training set
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute the cross validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")

        '''
        Training MSE: 49.11160933402521
        Cross validation MSE: 87.69841211111924
        '''
        ```

    - the MSEs are significantly better for both the training and cross validation set when the second order polynomial is added
    - introduce more polynomial terms and see which one gives the best performance
    - create a loop that contains all the steps in the previous code cells
    - here is one implementation that adds polynomial features up to degree=10

        ```python
        # Initialize lists to save the errors, models, and feature transforms
        train_mses = []
        cv_mses = []
        models = []
        polys = []
        scalers = []

        # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
        for degree in range(1,11):

            # Add polynomial features to the training set
            poly = PolynomialFeatures(degree, include_bias=False)
            X_train_mapped = poly.fit_transform(x_train)
            polys.append(poly)

            # Scale the training set
            scaler_poly = StandardScaler()
            X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
            scalers.append(scaler_poly)

            # Create and train the model
            model = LinearRegression()
            model.fit(X_train_mapped_scaled, y_train )
            models.append(model)

            # Compute the training MSE
            yhat = model.predict(X_train_mapped_scaled)
            train_mse = mean_squared_error(y_train, yhat) / 2
            train_mses.append(train_mse)

            # Add polynomial features and scale the cross validation set
            X_cv_mapped = poly.transform(x_cv)
            X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

            # Compute the cross validation MSE
            yhat = model.predict(X_cv_mapped_scaled)
            cv_mse = mean_squared_error(y_cv, yhat) / 2
            cv_mses.append(cv_mse)

        # Plot the results
        degrees=range(1,11)
        utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree of polynomial vs. train and CV MSEs")
        ```

- Choosing the best model
    - when selecting a model, you want to choose one that performs well both on the training and cross validation set
    - it implies that it is ale to lean the patterns from the training set without overfitting
    - there is a sharp drop in dev error from the models with degree=1 to degree=2
    - followed by a relatively flat line up to degree=5
    - the dev error than gets worse with more polynomial features
    - the model with the lowest cv_mse is chosen to the the one best suited for the application

        ```python
        # Get the model with the lowest CV MSE (add 1 because list indices start at 0)
        # This also corresponds to the degree of the polynomial added
        degree = np.argmin(cv_mses) + 1
        print(f"Lowest CV MSE is found in the model with degree={degree}")

        # Lowest CV MSE is found in the model with degree=4
        ```

    - then publish the generalization error by computing the test set’s MSE
    - transform the data the same way

        ```python
        # Add polynomial features to the test set
        X_test_mapped = polys[degree-1].transform(x_test)

        # Scale the test set
        X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

        # Compute the test MSE
        yhat = models[degree-1].predict(X_test_mapped_scaled)
        test_mse = mean_squared_error(y_test, yhat) / 2

        print(f"Training MSE: {train_mses[degree-1]:.2f}")
        print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
        print(f"Test MSE: {test_mse:.2f}")

        '''
        Training MSE: 47.15
        Cross Validation MSE: 79.43
        Test MSE: 104.63
        '''
        ```

Neural Networks

- the same model selection process can also be used when choosing between different NN architectures
- create the models shown below and apply them to the same regression task

- Prepare the data
    - use the same training, cross validation, and test sets generated in the previous section
    - NN can learn non-linear relationships so you can opt to skip adding polynomial features
    - the default degree is set to 1 to indicate that the model will just use x_train, x_cv, and x_test as is

        ```python
        # Add polynomial features
        degree = 1
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)
        X_cv_mapped = poly.transform(x_cv)
        X_test_mapped = poly.transform(x_test)
        ```

    - scale the input features to help GD converge faster
    - notice that the mean and standard deviation computed from the training set is used with transform() in the dev and test sets instead of fit_transform()

        ```python
        # Scale the features using the z-score
        scaler = StandardScaler()
        X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
        X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
        X_test_mapped_scaled = scaler.transform(X_test_mapped)
        ```

- Build and train the models
    - create the NN architectures
    - the code is provided in the build_models() function in the utils.py() file
    - for each model, also record the training and dev errors

        ```python
        # Initialize lists that will contain the errors for each model
        nn_train_mses = []
        nn_cv_mses = []

        # Build the models
        nn_models = utils.build_models()

        # Loop over the the models
        for model in nn_models:

            # Setup the loss and optimizer
            model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            )

            print(f"Training {model.name}...")

            # Train the model
            model.fit(
                X_train_mapped_scaled, y_train,
                epochs=300,
                verbose=0
            )

            print("Done!\n")

            # Record the training MSEs
            yhat = model.predict(X_train_mapped_scaled)
            train_mse = mean_squared_error(y_train, yhat) / 2
            nn_train_mses.append(train_mse)

            # Record the cross validation MSEs 
            yhat = model.predict(X_cv_mapped_scaled)
            cv_mse = mean_squared_error(y_cv, yhat) / 2
            nn_cv_mses.append(cv_mse)

        # print results
        print("RESULTS:")
        for model_num in range(len(nn_train_mses)):
            print(
                f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
                f"CV MSE: {nn_cv_mses[model_num]:.2f}"
                )

        '''
        Training model_1...
        Done!

        Training model_2...
        Done!

        Training model_3...
        Done!

        RESULTS:
        Model 1: Training MSE: 73.44, CV MSE: 113.87
        Model 2: Training MSE: 73.40, CV MSE: 112.28
        Model 3: Training MSE: 44.56, CV MSE: 88.51
        '''
        ```

    - from the recorded errors, decide which is the best model for the application
    - compute the test error to estimate how well the model generalizes to new examples

        ```python
        # Select the model with the lowest CV MSE
        model_num = 3

        # Compute the test MSE
        yhat = nn_models[model_num-1].predict(X_test_mapped_scaled)
        test_mse = mean_squared_error(y_test, yhat) / 2

        print(f"Selected Model: {model_num}")
        print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
        print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
        print(f"Test MSE: {test_mse:.2f}")

        '''
        Selected Model: 3
        Training MSE: 44.56
        Cross Validation MSE: 88.51
        Test MSE: 87.77
        '''
        ```

Classification

- practice model evaluation and selection on a classification task
- similar, with the main difference being the computation of errors
- Load the Dataset
    - load a dataset for a binary classification task
    - has 200 examples of two input features (x1 and x2), and a target y of either 0 or 1

        ```python
        # Load the dataset from a text file
        data = np.loadtxt('./data/data_w3_ex2.csv', delimiter=',')

        # Split the inputs and outputs into separate arrays
        x_bc = data[:,:-1]
        y_bc = data[:,-1]

        # Convert y into 2-D because the commands later will require it (x is already 2-D)
        y_bc = np.expand_dims(y_bc, axis=1)

        print(f"the shape of the inputs x is: {x_bc.shape}")
        print(f"the shape of the targets y is: {y_bc.shape}")

        '''
        the shape of the inputs x is: (200, 2)
        the shape of the targets y is: (200, 1)
        '''
        ```

- Split and prepare the dataset
    - generate the training, cross validation, and test sets
    - use the same 60/20/20 proportions
    - scale the features as in the previous sections

        ```python
        from sklearn.model_selection import train_test_split

        # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
        x_bc_train, x_, y_bc_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

        # Split the 40% subset above into two: one half for cross validation and the other for the test set
        x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

        # Delete temporary variables
        del x_, y_

        print(f"the shape of the training set (input) is: {x_bc_train.shape}")
        print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
        print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
        print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
        print(f"the shape of the test set (input) is: {x_bc_test.shape}")
        print(f"the shape of the test set (target) is: {y_bc_test.shape}")

        '''
        the shape of the training set (input) is: (120, 2)
        the shape of the training set (target) is: (120, 1)

        the shape of the cross validation set (input) is: (40, 2)
        the shape of the cross validation set (target) is: (40, 1)

        the shape of the test set (input) is: (40, 2)
        the shape of the test set (target) is: (40, 1)
        '''
        ```

        ```python
        # Scale the features

        # Initialize the class
        scaler_linear = StandardScaler()

        # Compute the mean and standard deviation of the training set then transform it
        x_bc_train_scaled = scaler_linear.fit_transform(x_bc_train)
        x_bc_cv_scaled = scaler_linear.transform(x_bc_cv)
        x_bc_test_scaled = scaler_linear.transform(x_bc_test)
        ```

- Evaluating the error for classification models
    - in previous sections on regression models, the mean squared error was used to measure the accuracy of the model
    - for classification, a similar metric by getting the fraction of the data that the model has misclassified is used
    - if you model made wrong predictions for 2 samples out of 5, then you will report an error of 40% or 0.4

        ```python
        # Sample model output
        probabilities = np.array([0.2, 0.6, 0.7, 0.3, 0.8])

        # Apply a threshold to the model output. If greater than 0.5, set to 1. Else 0.
        predictions = np.where(probabilities >= 0.5, 1, 0)

        # Ground truth labels
        ground_truth = np.array([1, 1, 1, 1, 1])

        # Initialize counter for misclassified data
        misclassified = 0

        # Get number of predictions
        num_predictions = len(predictions)

        # Loop over each prediction
        for i in range(num_predictions):

            # Check if it matches the ground truth
            if predictions[i] != ground_truth[i]:

                # Add one to the counter if the prediction is wrong
                misclassified += 1

        # Compute the fraction of the data that the model misclassified
        fraction_error = misclassified/num_predictions

        print(f"probabilities: {probabilities}")
        print(f"predictions with threshold=0.5: {predictions}")
        print(f"targets: {ground_truth}")
        print(f"fraction of misclassified data (for-loop): {fraction_error}")
        print(f"fraction of misclassified data (with np.mean()): {np.mean(predictions != ground_truth)}")

        '''
        probabilities: [0.2 0.6 0.7 0.3 0.8]
        predictions with threshold=0.5: [0 1 1 0 1]
        targets: [1 1 1 1 1]
        fraction of misclassified data (for-loop): 0.4
        fraction of misclassified data (with np.mean()): 0.4
        '''
        ```

- Build and train the model
    - use the same NN architecture in the previous section, build_models() function can be called again to create new instances of the models
    - follow the recommended approach with linear activation for the output (instead of sigmoid) then set from_logits=true when declaring the loss function of the model
    - use the binary crossentropy loss for binart classification
    - after training, use a sigmoid function to convert the model outputs into probabilities
    - set a threshold and get the fraction of misclassified examples from the training and cross validation sets

        ```python
        # Initialize lists that will contain the errors for each model
        nn_train_error = []
        nn_cv_error = []

        # Build the models
        models_bc = utils.build_models()

        # Loop over each model
        for model in models_bc:

            # Setup the loss and optimizer
            model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            )

            print(f"Training {model.name}...")

            # Train the model
            model.fit(
                x_bc_train_scaled, y_bc_train,
                epochs=200,
                verbose=0
            )

            print("Done!\n")

            # Set the threshold for classification
            threshold = 0.5

            # Record the fraction of misclassified examples for the training set
            yhat = model.predict(x_bc_train_scaled)
            yhat = tf.math.sigmoid(yhat)
            yhat = np.where(yhat >= threshold, 1, 0)
            train_error = np.mean(yhat != y_bc_train)
            nn_train_error.append(train_error)

            # Record the fraction of misclassified examples for the cross validation set
            yhat = model.predict(x_bc_cv_scaled)
            yhat = tf.math.sigmoid(yhat)
            yhat = np.where(yhat >= threshold, 1, 0)
            cv_error = np.mean(yhat != y_bc_cv)
            nn_cv_error.append(cv_error)

        # Print the result
        for model_num in range(len(nn_train_error)):
            print(
                f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
                f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
                )

        '''
        Training model_1...
        Done!

        Training model_2...
        Done!

        Training model_3...
        Done!

        Model 1: Training Set Classification Error: 0.05833, CV Set Classification Error: 0.17500
        Model 2: Training Set Classification Error: 0.06667, CV Set Classification Error: 0.15000
        Model 3: Training Set Classification Error: 0.05000, CV Set Classification Error: 0.15000
        '''
        ```

    - from the output above, choose which model performed best
    - tie on the cross validation set error, then you can add another criteria to break it
    - a more common approach is to choose the smaller model because it saves computational resources

        ```python
        # Select the model with the lowest error
        model_num = 3

        # Compute the test error
        yhat = models_bc[model_num-1].predict(x_bc_test_scaled)
        yhat = tf.math.sigmoid(yhat)
        yhat = np.where(yhat >= threshold, 1, 0)
        nn_test_error = np.mean(yhat != y_bc_test)

        print(f"Selected Model: {model_num}")
        print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
        print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
        print(f"Test Set Classification Error: {nn_test_error:.4f}")

        '''
        Selected Model: 3
        Training Set Classification Error: 0.0500
        CV Set Classification Error: 0.1500
        Test Set Classification Error: 0.1750
        '''
        ```