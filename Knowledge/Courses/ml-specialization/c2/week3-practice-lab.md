Packages

- import packages

    ```python
    import numpy as np
    %matplotlib widget
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.activations import relu,linear
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from tensorflow.keras.optimizers import Adam

    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    from public_tests_a1 import * 

    tf.keras.backend.set_floatx('float64')
    from assigment_utils import *

    tf.autograph.set_verbosity(0)
    ```

Evaluating a Learning Alg (Polynomial Regression)

- goal of creating a model is to be able to predict values for new examples
- to test the model’s performance on new data before deploying
    - split original data set into ‘training’ and ‘test’ sets
        - use training data to fit the parameters of the model
        - use test data to evaluate the model on new data
    - develop an error function to evaluate the model
- Splitting the data set
    - reserve 20-40% of data set for testing
    - use sklearn function train_test_split to perform the split
    - double-chheck the shapes after running the following cell

        ```python
        # Generate some data
        X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
        print("X.shape", X.shape, "y.shape", y.shape)

        #split the data using sklearn routine 
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)
        print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
        print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

        '''
        X.shape (18,) y.shape (18,)
        X_train.shape (12,) y_train.shape (12,)
        X_test.shape (6,) y_test.shape (6,)
        '''
        ```

- Plot Train, Test sets
    - the data point that will be part of training (in red) are intermixed with those that the model is not trained on (test)
    - this particular data set is a quadratic function with noise added
    - ‘ideal’ curve below

        ```python
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
        ax.set_title("Training, Test",fontsize = 14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.scatter(X_train, y_train, color = "red",           label="train")
        ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
        ax.legend(loc='upper left')
        plt.show()
        ```

- Error calculation for model evaluation, linear regression
    - when evaluating a linear regression model, you average the squared error difference of the predicted values and the target values

- Exercise 1
    - create a function to evaluate the error on a data set for a linear regression model

        ```python
        # UNQ_C1
        # GRADED CELL: eval_mse
        def eval_mse(y, yhat):
            """ 
            Calculate the mean squared error on a data set.
            Args:
              y    : (ndarray  Shape (m,) or (m,1))  target value of each example
              yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
            Returns:
              err: (scalar)             
            """
            m = len(y)
            err = 0.0
            for i in range(m):
            ### START CODE HERE ### 

                err += (y[i] - yhat[i])**2

            err /= (2 * m)

            ### END CODE HERE ### 

            return(err)
        ```

- Compare performance on training and test data
    - build a high degree polynomial model to minimize training
    - use the linear_regression functions from sklearn
        - create and fit the model
        - compute the error on the training data
        - compute the error on the test data

            ```python
            # create a model in sklearn, train on training data
            degree = 10
            lmodel = lin_model(degree)
            lmodel.fit(X_train, y_train)

            # predict on training data, find training error
            yhat = lmodel.predict(X_train)
            err_train = lmodel.mse(y_train, yhat)

            # predict on test data, find error
            yhat = lmodel.predict(X_test)
            err_test = lmodel.mse(y_test, yhat)
            ```

            ```python
            print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

            # training err 58.01, test err 171215.01
            ```

    - the computed error on the training set is substantially less than that of the test set
    - the model fits the training data very well
    - to do so, it has created a complex function
    - the test data was not part of the training and the model does a poor job of predicting on this data
    - this model would be described as 1) overfitting, 2) high variance 3) generalizes poorly

        ```python
        # plot predictions over data range 
        x = np.linspace(0,int(X.max()),100)  # predict values for plot
        y_pred = lmodel.predict(x).reshape(-1,1)

        plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)
        ```

    - the test set error shows this model will not work well on new data
    - if you use the test error to guide improvements in the model, then the model will perform well on the test data, but the test data was meant to represent new data
    - you need yet another set of data to test new data performance
    - separate data into three groups
    - the distribution of training, cross-validation and test sets is a typical distribution, but can be varied depending on the amount of data available

    - generate three data sets
    - use train_test_split from sklearn but call it twice to get three splits

        ```python
        # Generate  data
        X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
        print("X.shape", X.shape, "y.shape", y.shape)

        #split the data using sklearn routine 
        X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
        X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)
        print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
        print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
        print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

        '''
        X.shape (40,) y.shape (40,)
        X_train.shape (24,) y_train.shape (24,)
        X_cv.shape (8,) y_cv.shape (8,)
        X_test.shape (8,) y_test.shape (8,)
        '''
        ```

- Bias and Variance
    - clear that the degree of the polynomial model was too high
    - the training and cross-validation performance can provide guidance to choose a good value
    - by trying a range of degree values, the training and cross-validation performance can be evaluated
    - as the degree becomes too large, the cross-validation performance will start to degrade relative to the training performance
    - Plot Train, Cross-Validation, Test
        - see the datapoints that will be part of training (in red) are intermixed with those that the model is not trained on test and cv)

            ```python
            fig, ax = plt.subplots(1,1,figsize=(4,4))
            ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
            ax.set_title("Training, CV, Test",fontsize = 14)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            ax.scatter(X_train, y_train, color = "red",           label="train")
            ax.scatter(X_cv, y_cv,       color = dlc["dlorange"], label="cv")
            ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
            ax.legend(loc='upper left')
            plt.show()
            ```

    - Finding the optimal degree
        - train the model repeatedly, increasing the degree of the polynomial each iteration
        - use the scikit-learn linear regression model for speed and simpicity

            ```python
            max_degree = 9
            err_train = np.zeros(max_degree)    
            err_cv = np.zeros(max_degree)      
            x = np.linspace(0,int(X.max()),100)  
            y_pred = np.zeros((100,max_degree))  #columns are lines to plot

            for degree in range(max_degree):
                lmodel = lin_model(degree+1)
                lmodel.fit(X_train, y_train)
                yhat = lmodel.predict(X_train)
                err_train[degree] = lmodel.mse(y_train, yhat)
                yhat = lmodel.predict(X_cv)
                err_cv[degree] = lmodel.mse(y_cv, yhat)
                y_pred[:,degree] = lmodel.predict(x)

            optimal_degree = np.argmin(err_cv)+1
            ```

            ```python
            plt.close("all")
            plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal, 
                               err_train, err_cv, optimal_degree, max_degree)
            ```

        - the plot demonstrates that separating data into two groups, data the model is trained on and data the model has not been trained on, can be used to determine if the model is underfitting or overfitting
        - in the example, a variety of models varying from underfitting to overfitting by increasing the degree of the polynomial used
        - on the left plot, the solid lines represent the predictions from these models
        - a polynomial model with degree 1 produces a straight line that intersects very few data points, while the maximum degree hews very closely to every data point
        - on the right
            - the error on the trained data (blue) decreases as the model complexity increases as expected
            - the error of the cross-validation data decreases initially as the model starts to conform to the data, but then increases as the model starts to over-fit on the training data (fails to generalize)
        - worth noting that the curves in the examples are not smooth as one might draw for a lecture
        - it’s clear that the specific data points assigned to each group can change the results significantly
        - the general trend is what is important
    - Tuning Regularization
        - start with a high degree polynomial and vary the regularization parameter

            ```python
            lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
            num_steps = len(lambda_range)
            degree = 10
            err_train = np.zeros(num_steps)    
            err_cv = np.zeros(num_steps)       
            x = np.linspace(0,int(X.max()),100) 
            y_pred = np.zeros((100,num_steps))  #columns are lines to plot

            for i in range(num_steps):
                lambda_= lambda_range[i]
                lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
                lmodel.fit(X_train, y_train)
                yhat = lmodel.predict(X_train)
                err_train[i] = lmodel.mse(y_train, yhat)
                yhat = lmodel.predict(X_cv)
                err_cv[i] = lmodel.mse(y_cv, yhat)
                y_pred[:,i] = lmodel.predict(x)

            optimal_reg_idx = np.argmin(err_cv) 
            ```

            ```python
            plt.close("all")
            plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)
            ```

        - the plots show that as regularization increases, the model moves from a high variance model to a high bias model
    - Getting more data: Increasing Training Set Size (m)
        - when a model is overfitting, collecting additional data can improve performance

            ```python
            X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
            plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)
            ```

        - the above plots show that when a model has high variance and is overfitting, addingmore examples improves performance
        - the curves on the left plot with the highest value of m is a smooth curve that is in the center of the data
        - on the right, as the number of examples increases, the performance of the training set and cross-validation set converge to similar values
        - adding more examples when the model has high bias does not improve performance
- Evaluating a Learning Alg (NN)
    - Data Set
        - generate a data set and split it into training, cross-validation and test sets

            ```python
            # Generate and split data set
            X, y, centers, classes, std = gen_blobs()

            # split the data. Large CV population for demonstration
            X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
            X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)
            print("X_train.shape:", X_train.shape, "X_cv.shape:", X_cv.shape, "X_test.shape:", X_test.shape)
            ```

            ```python
            plt_train_eq_dist(X_train, y_train,classes, X_cv, y_cv, centers, std)
            ```

        - there are six clusters identified by color
        - both training points (dots) and CV points (triangle) are shown
        - the interesting points are those that fall in ambiguous locations where either cluster might consider them members
        - on the right is an example of an ‘ideal’ model, or a model one might create knowing the source of the data
        - the lines represent ‘equal distance’ boundaries where the distance between center points is equal
        - this model would ‘misclassify’ roughly 8% of the total data set
    - Evaluating categorical model by calculating classification error
        - the eval function for categorical models used here is simply the fraction of incorrect predictions

- Exercise 2
    - complete the routine to calculate classification error
    - target values are the index of the category and are not one-hot encoded

        ```python
        # UNQ_C2
        # GRADED CELL: eval_cat_err
        def eval_cat_err(y, yhat):
            """ 
            Calculate the categorization error
            Args:
              y    : (ndarray  Shape (m,) or (m,1))  target value of each example
              yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
            Returns:|
              cerr: (scalar)             
            """
            m = len(y)
            incorrect = 0
            for i in range(m):
            ### START CODE HERE ### 

                if y[i] != yhat[i]:
                    incorrect += 1

            cerr = incorrect / m

            ### END CODE HERE ### 

            return(cerr)
        ```

        ```python
        y_hat = np.array([1, 2, 0])
        y_tmp = np.array([1, 2, 3])
        print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333" )
        y_hat = np.array([[1], [2], [0], [3]])
        y_tmp = np.array([[1], [2], [1], [3]])
        print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250" )

        # BEGIN UNIT TEST  
        test_eval_cat_err(eval_cat_err)
        # END UNIT TEST

        '''
        categorization error 0.333, expected:0.333
        categorization error 0.250, expected:0.250
         All tests passed.
        '''
        ```

- Model Complexity
    - build two models
    - a complex model and a simple model
    - evaluate the models to determine if they are likely to overfit or underfit
    - Complex model
    - Exercise 3
        - compose a three-layer model:
            - dense layer with 120 units, relu activation
            - dense layer with 40 units, relu activation
            - dense layer with 6 units and a linear activation (not softmax)
            - loss with SparseCategoricalCrossentropy
            - from_logits=True
            - Adam optimizer with learning rate of 0.01

            ```python
            # UNQ_C3
            # GRADED CELL: model
            import logging
            logging.getLogger("tensorflow").setLevel(logging.ERROR)

            tf.random.set_seed(1234)
            model = Sequential(
                [
                    ### START CODE HERE ### 

                    Dense(units=120, activation='relu'),
                    Dense(units=40, activation='relu'),
                    Dense(units=6, activation='linear')

                    ### END CODE HERE ### 

                ], name="Complex"
            )
            model.compile(
                ### START CODE HERE ### 
                loss=SparseCategoricalCrossentropy(from_logits=True),
                optimizer=Adam(0.01),
                ### END CODE HERE ### 
            )
            ```

            ```python
            # BEGIN UNIT TEST
            model.fit(
                X_train, y_train,
                epochs=1000
            )
            # END UNIT TEST

            '''
            Epoch 1/1000
            13/13 [==============================] - 0s 1ms/step - loss: 1.1106
            Epoch 2/1000
            13/13 [==============================] - 0s 1ms/step - loss: 0.4281
            ...
            Epoch 999/1000
            13/13 [==============================] - 0s 3ms/step - loss: 0.0155
            Epoch 1000/1000
            13/13 [==============================] - 0s 1ms/step - loss: 0.0172
            '''
            ```

            ```python
            # BEGIN UNIT TEST
            model.summary()

            model_test(model, classes, X_train.shape[1]) 
            # END UNIT TEST

            '''
            Model: "Complex"
            _________________________________________________________________
             Layer (type)                Output Shape              Param #   
            =================================================================
             dense_24 (Dense)            (None, 120)               360       

             dense_25 (Dense)            (None, 40)                4840      

             dense_26 (Dense)            (None, 6)                 246       

            =================================================================
            Total params: 5,446
            Trainable params: 5,446
            Non-trainable params: 0
            _________________________________________________________________
            All tests passed!
            '''
            ```

            ```python
            #make a model for plotting routines to call
            model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
            plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")
            ```

        - the model worked very hard to capture outliers of each category
        - as a result, it has miscategorized some of the cross-validation data

            ```python
            training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
            cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
            print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
            print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")

            '''
            categorization error, training, complex model: 0.003
            categorization error, cv,       complex model: 0.122
            '''
            ```

    - Simple model
    - Exercise 4
        - compose a two-layer model:
            - dense layer with 6 units, relu
            - dense layer with 6 units, linear
            - SparseCategoricalCrossentropy
            - from_logits=Ture
            - Adam optimizer, 0.01

            ```python
            # UNQ_C4
            # GRADED CELL: model_s

            tf.random.set_seed(1234)
            model_s = Sequential(
                [
                    ### START CODE HERE ### 

                    Dense(units=6, activation='relu'),
                    Dense(units=6, activation='linear')

                    ### END CODE HERE ### 
                ], name = "Simple"
            )
            model_s.compile(
                ### START CODE HERE ### 
                loss=SparseCategoricalCrossentropy(from_logits=True),
                optimizer=Adam(0.01),
                ### START CODE HERE ### 
            )
            ```

            ```python
            import logging
            logging.getLogger("tensorflow").setLevel(logging.ERROR)

            # BEGIN UNIT TEST
            model_s.fit(
                X_train,y_train,
                epochs=1000
            )
            # END UNIT TEST

            '''
            Epoch 1/1000
            13/13 [==============================] - 0s 952us/step - loss: 1.7306
            Epoch 2/1000
            13/13 [==============================] - 0s 874us/step - loss: 1.4468
            ...
            Epoch 999/1000
            13/13 [==============================] - 0s 840us/step - loss: 0.1711
            Epoch 1000/1000
            13/13 [==============================] - 0s 830us/step - loss: 0.1628
            '''
            ```

            ```python
            # BEGIN UNIT TEST
            model_s.summary()

            model_s_test(model_s, classes, X_train.shape[1])
            # END UNIT TEST

            '''
            Model: "Simple"
            _________________________________________________________________
             Layer (type)                Output Shape              Param #   
            =================================================================
             dense_27 (Dense)            (None, 6)                 18        

             dense_28 (Dense)            (None, 6)                 42        

            =================================================================
            Total params: 60
            Trainable params: 60
            Non-trainable params: 0
            _________________________________________________________________
            All tests passed!
            '''
            ```

            ```python
            #make a model for plotting routines to call
            model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(),axis=1)
            plt_nn(model_predict_s,X_train,y_train, classes, X_cv, y_cv, suptitle="Simple Model")
            ```

            ```python
            training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
            cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
            print(f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
            print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )

            '''
            categorization error, training, simple model, 0.062, complex model: 0.003
            categorization error, cv,       simple model, 0.087, complex model: 0.122
            '''
            ```

        - the simple model has a higher classification error on training data but does better on CV data than the more complex model
- Regularization
    - apply regularization to moderate the impact of a more complex model
    - Exercise 5
        - reconstruct the complex model, but include regularization
        - compose a three-layer model
            - dense layer with 120 units, relu, kernel_regularizer=tf.karas.regularizers.l2(0,1)
            - dense layer with 40 units, relu, same regularizer
            - dense layer with 6 units, linear
            - Sparse.., from_logits=.., Adam..

            ```python
            # UNQ_C5
            # GRADED CELL: model_r

            tf.random.set_seed(1234)
            model_r = Sequential(
                [
                    ### START CODE HERE ### 

                    Dense(units=120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
                    Dense(units=40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
                    Dense(units=6, activation='linear')

                    ### START CODE HERE ### 
                ], name= None
            )
            model_r.compile(
                ### START CODE HERE ### 
                loss=SparseCategoricalCrossentropy(from_logits=True),
                optimizer=Adam(0.01),
                ### START CODE HERE ### 
            )
            ```

            ```python
            # BEGIN UNIT TEST
            model_r.fit(
                X_train, y_train,
                epochs=1000
            )
            # END UNIT TEST

            '''
            Epoch 1/1000
            13/13 [==============================] - 0s 1ms/step - loss: 4.4464
            Epoch 2/1000
            13/13 [==============================] - 0s 1ms/step - loss: 1.7086
            ...
            Epoch 999/1000
            13/13 [==============================] - 0s 1ms/step - loss: 0.3505
            Epoch 1000/1000
            13/13 [==============================] - 0s 3ms/step - loss: 0.3514
            '''
            ```

            ```python
            # BEGIN UNIT TEST
            model_r.summary()

            model_r_test(model_r, classes, X_train.shape[1]) 
            # END UNIT TEST

            '''
            Model: "sequential_1"
            _________________________________________________________________
             Layer (type)                Output Shape              Param #   
            =================================================================
             dense_32 (Dense)            (None, 120)               360       

             dense_33 (Dense)            (None, 40)                4840      

             dense_34 (Dense)            (None, 6)                 246       

            =================================================================
            Total params: 5,446
            Trainable params: 5,446
            Non-trainable params: 0
            _________________________________________________________________
            ddd
            All tests passed!
            '''
            ```

            ```python
            #make a model for plotting routines to call
            model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(),axis=1)

            plt_nn(model_predict_r, X_train,y_train, classes, X_cv, y_cv, suptitle="Regularized")
            ```

            ```python
            training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
            cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
            test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
            print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
            print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )

            '''
            categorization error, training, regularized: 0.072, simple model, 0.062, complex model: 0.003
            categorization error, cv,       regularized: 0.066, simple model, 0.087, complex model: 0.122
            '''
            ```

        - the simple model is a bit better in the training set than the regularized model but worse in the cross validation set
- Iterate to find optimal regularization value
    - try many regularization values

        ```python
        tf.random.set_seed(1234)
        lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
        models=[None] * len(lambdas)

        for i in range(len(lambdas)):
            lambda_ = lambdas[i]
            models[i] =  Sequential(
                [
                    Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                    Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                    Dense(classes, activation = 'linear')
                ]
            )
            models[i].compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(0.01),
            )

            models[i].fit(
                X_train,y_train,
                epochs=1000
            )
            print(f"Finished lambda = {lambda_}")

        '''
        Epoch 1/1000
        13/13 [==============================] - 0s 1ms/step - loss: 1.1106
        ...
        Epoch 1000/1000
        13/13 [==============================] - 0s 1ms/step - loss: 0.0172
        Finished lambda = 0.0
        Epoch 1/1000
        13/13 [==============================] - 0s 1ms/step - loss: 1.1055
        ...
        Epoch 1000/1000
        13/13 [==============================] - 0s 1ms/step - loss: 0.1538
        Finished lambda = 0.001
        Epoch 1/1000
        13/13 [==============================] - 0s 1ms/step - loss: 1.4887
        ...
        Epoch 1000/1000
        13/13 [==============================] - 0s 1ms/step - loss: 0.2322
        Finished lambda = 0.01
        Epoch 1/1000
        13/13 [==============================] - 0s 1ms/step - loss: 3.0747
        ...
        Epoch 1000/1000
        13/13 [==============================] - 0s 1ms/step - loss: 0.3030
        Finished lambda = 0.05
        Epoch 1/1000
        13/13 [==============================] - 0s 1ms/step - loss: 4.3818
        ...
        Epoch 1000/1000
        13/13 [==============================] - 0s 3ms/step - loss: 0.3561
        Finished lambda = 0.1
        Epoch 1/1000
        13/13 [==============================] - 0s 1ms/step - loss: 7.3305
        ...
        Epoch 1000/1000
        13/13 [==============================] - 0s 3ms/step - loss: 0.3814
        Finished lambda = 0.2
        Epoch 1/1000
        13/13 [==============================] - 0s 1ms/step - loss: 9.8240
        ...
        Epoch 1000/1000
        13/13 [==============================] - 0s 3ms/step - loss: 0.4581
        Finished lambda = 0.3
        '''
        ```

        ```python
        plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)
        ```

    - as regularization is increased, the performance of the model on the training and CV data sets converge
    - for this data set and model, lambda > 0.01 seems to be a reasonable choice
- Test
    - try the optimized models on the test set and compare them to ‘ideal’ performance

    - the test set is small and seems to have a number of outliers so classification error is high
    - however, the performance of the optimized models is comparable to ideal performance