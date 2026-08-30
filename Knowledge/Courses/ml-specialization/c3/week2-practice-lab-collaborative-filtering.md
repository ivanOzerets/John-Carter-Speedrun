Packages

- numpy and tensorflow

    ```python
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from recsys_utils import *
    ```

Notation

- list of notation

Recommender Systems

- implement the collaborative filtering learning alg and apply it to a dataset of movie ratings
- the goal of a collaborative filtering recommender system is to generate two vectors
- for each user, a ‘parameter vector’ that embodies the movie tastes of a user
- for each movie, a feature vector of the same size which embodies some description of the movie
- the dot product of the two vectors plus the bias term should produce an estimate of the rating the user might give to that movie

- existing rating are provided in matrix form as shown
- Y contains ratings; 0.5 to 5 inclusive in 0.5 steps
- 0 if the movie has not been rated
- R has a 1 where movies have been rated
- movies are in rows, users in columns
- each user has a parameter vector w^user and bias
- each movie has a feature vector x^movie
- these vectors are simultaneously learned by using the existing user/movie ratings as training data
- one training example is shown above: w^1 . x^1 + b^1 = 4
- it is worth noting that the feature vector x^movie must satisfy all the users while the user vector w^user must satisfy all the movies
- this is the source of the name of this approach - all the users collaborate to generate the rating set

- once the feature vectors and parameters are learned, they can be used to predict how a user might rate an unrated movie
- the equation is an example of predicting a rating for user one on movie zero
- implement the function cofiCostFunc that computes the collaborative filtering objective function
- after implementing the objective function, use a tensorflow custom training loop to learn the parameters for collaborative filtering
- the first step is to detail the data set and data structures thatwill be used in the lab

Movie ratings dataset

- data set is derived from the [MovieLens "ml-latest-small"](https://grouplens.org/datasets/movielens/latest/) dataset
- the original dataset has 9000 movies rated by 600 users
- the dataset has been reduced in size to focus on movies from the years since 2000
- this dataset consists of ratings on a scale of 0.5 to 5 in 0.5 step increments
- the reduced dataset has n_u = 443 users, and n_m = 4779 movies
- below, you will load the movie dataset into the variables Y and R
- the matrix Y (n_m x n_u) stores the ratings y^(i,j)
- the matrix R is a binary-valued indicator matrix, where R(i,j) = 1 if user j gave a rating to movie i, and R(i,j) = 0 otherwise

- the ith row of X corresponds to the feature vector x^i for the ith movie, and the jth row of W corresponds to one parameter vector w^j, for the jth user
- both x^i and w^j are n-dimensional vectors
- use n = 10, and therefore, x^i and w^j have 10 elements
- correspondingly, X is a n_m x 10 matrix and W is a n_u x 10 matrix
- start by loading the move ratings dataset to understand the structure of the data
- load Y and R with the movie dataset
- load X, W, and b with pre-computed values
- these values will be learned later in the lab

    ```python
    #Load data
    X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
    Y, R = load_ratings_small()

    print("Y", Y.shape, "R", R.shape)
    print("X", X.shape)
    print("W", W.shape)
    print("b", b.shape)
    print("num_features", num_features)
    print("num_movies",   num_movies)
    print("num_users",    num_users)

    '''
    Y (4778, 443) R (4778, 443)
    X (4778, 10)
    W (443, 10)
    b (1, 443)
    num_features 10
    num_movies 4778
    num_users 443
    '''
    ```

    ```python
    #  From the matrix, we can compute statistics like average rating.
    tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
    print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )

    # Average rating for movie 1 : 3.400 / 5
    ```

Collaborative filtering learning alg

- begin implementing the collaborative filtering learning alg
- start by implementing the objective function
- the collaborative filtering alg in the setting of movie recommendations considers a set of n-dimensional parameter vectors x^0, … x^(n_m - 1) and b^0, …, b^(n_u - 1), where the model predicts the rating for movie i by user j as y^(i,j) = w^j . x^i + b^j
- given a dataset that consists of a set of ratings produced by some users on some movies, you with to learn the parameter vectors x^0, … x^(n_m - 1) and b^0, …, b^(n_u - 1) that produce the best fit (minimizes the squared error)
- complete the code in cofiCostFunc to compute the cost function for collaborative filtering
- Collaborative filtering cost function
    - the collaborative filtering cost function is given by

    - the first summation is “for all i, j where r(i,j) equals 1 and could be written

- Exercise 1
    - For loop implementation
        - start by implementing the cost function using for loops
        - consider developing the cost function in two steps
        - first, develop the cost function without regularization
        - a test case that does not include regularization is provided below to test your implementation
        - then, add regularization and run the tests that include regularization
        - note that accumulating the cost for user j and movie i only if R(i,j) = 1

        ```python
        # GRADED FUNCTION: cofi_cost_func
        # UNQ_C1

        def cofi_cost_func(X, W, b, Y, R, lambda_):
            """
            Returns the cost for the content-based filtering
            Args:
              X (ndarray (num_movies,num_features)): matrix of item features
              W (ndarray (num_users,num_features)) : matrix of user parameters
              b (ndarray (1, num_users)            : vector of user parameters
              Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
              R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
              lambda_ (float): regularization parameter
            Returns:
              J (float) : Cost
            """
            nm, nu = Y.shape
            J = 0
            ### START CODE HERE ###  

            for j in range (nu):
                w = W[j, :]
                b_j = b[0,j]
                for i in range (nm):
                    r = R[i,j]
                    x = X[i]
                    y = Y[i,j]
                    J += r * ((np.dot(w,x) + b_j - y) ** 2)

            J /= 2
            J += (lambda_/2) * (np.sum(W ** 2) + np.sum(X ** 2))

            ### END CODE HERE ### 

            return J
        ```

        ```python
        # Reduce the data set size so that this runs faster
        num_users_r = 4
        num_movies_r = 5 
        num_features_r = 3

        X_r = X[:num_movies_r, :num_features_r]
        W_r = W[:num_users_r,  :num_features_r]
        b_r = b[0, :num_users_r].reshape(1,-1)
        Y_r = Y[:num_movies_r, :num_users_r]
        R_r = R[:num_movies_r, :num_users_r]

        # Evaluate cost function
        J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0);
        print(f"Cost: {J:0.2f}")

        # Cost: 13.67
        ```

        ```python
        # Evaluate cost function with regularization 
        J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5);
        print(f"Cost (with regularization): {J:0.2f}")

        # Cost (with regularization): 28.09
        ```

    - Vectorized Implementation
        - important to create a vectorized implementation to compute J, since it will late be called many times during optimization
        - the linear algebra utilized is not the focus, so the implementation is provided

            ```python
            def cofi_cost_func_v(X, W, b, Y, R, lambda_):
                """
                Returns the cost for the content-based filtering
                Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
                Args:
                  X (ndarray (num_movies,num_features)): matrix of item features
                  W (ndarray (num_users,num_features)) : matrix of user parameters
                  b (ndarray (1, num_users)            : vector of user parameters
                  Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
                  R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
                  lambda_ (float): regularization parameter
                Returns:
                  J (float) : Cost
                """
                j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
                J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
                return J
            ```

            ```python
            # Evaluate cost function
            J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 0);
            print(f"Cost: {J:0.2f}")

            # Evaluate cost function with regularization 
            J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 1.5);
            print(f"Cost (with regularization): {J:0.2f}")

            '''
            Cost: 13.67
            Cost (with regularization): 28.09
            '''
            ```

Learning movie recommendations

- after you have finished implementing the collaborative filtering cost function, you can start training your alg to make movie recommendations
- in the cell below, enter movie choices
- the alg will then make recommendations

    ```python
    movieList, movieList_df = load_Movie_List_pd()

    my_ratings = np.zeros(num_movies)          #  Initialize my ratings

    # Check the file small_movie_list.csv for id of each movie in our dataset
    # For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
    my_ratings[2700] = 5 

    #Or suppose you did not enjoy Persuasion (2007), you can set
    my_ratings[2609] = 2;

    # We have selected a few movies we liked / did not like and the ratings we
    # gave are as follows:
    my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
    my_ratings[246]  = 5   # Shrek (2001)
    my_ratings[2716] = 3   # Inception
    my_ratings[1150] = 5   # Incredibles, The (2004)
    my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
    my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
    my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
    my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
    my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
    my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

    print('\nNew user ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0 :
            print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');

    '''
    New user ratings:

    Rated 2.0 for  Shrek (2001)
    Rated 5.0 for  Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    Rated 1.0 for  Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)
    Rated 5.0 for  Harry Potter and the Chamber of Secrets (2002)
    Rated 5.0 for  Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    Rated 5.0 for  Lord of the Rings: The Return of the King, The (2003)
    Rated 3.0 for  Eternal Sunshine of the Spotless Mind (2004)
    Rated 5.0 for  Incredibles, The (2004)
    Rated 2.0 for  Persuasion (2007)
    Rated 5.0 for  Toy Story 3 (2010)
    Rated 5.0 for  Inception (2010)
    Rated 1.0 for  Louis Theroux: Law & Disorder (2008)
    Rated 1.0 for  Nothing to Declare (Rien à déclarer) (2010)
    '''
    ```

- Add the reviews to Y and R and normalize the ratings

    ```python
    # Reload ratings
    Y, R = load_ratings_small()

    # Add new user ratings to Y 
    Y = np.c_[my_ratings, Y] # c_ is so special matrix concatenation

    # Add new user indicator matrix to R
    R = np.c_[(my_ratings != 0).astype(int), R]

    # Normalize the Dataset
    Ynorm, Ymean = normalizeRatings(Y, R)
    ```

- prepare to train the model
- initialize the parameters and select the Adam optimizer

    ```python
    #  Useful Values
    num_movies, num_users = Y.shape
    num_features = 100

    # Set Initial Parameters (W, X), use tf.Variable to track these variables
    tf.random.set_seed(1234) # for consistent results
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)
    ```

- train the collaborative filtering model
- this will learn the parameters X, W, and b
- the operations involved in learning w, b and x simultaneously do not fall into the typical ‘layers’ offered in the tensorflow neural network package
- consequently, the flow of model, compile, fit, predict are not directly applicable
- instead, use a custom training loop
- recall from earlier labs the steps of gradient descent
    - repeat until convergence
        - compute forward pass
        - compute the derivatives of the loss relative to parameters
        - update the parameters using the learning rate and the computed derivatives
- tensortflow has the marvelous capability of calculating the derivatives
- within the tf.gradienttape() section, operations on tensorflow variables are tracked
- when tape.gradient is later called, it will return the gradient of the loss relative to the tracked variables
- the gradients can then be applied to the parameters using an optimizer

    ```python
    iterations = 200
    lambda_ = 1
    for iter in range(iterations):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost 
        with tf.GradientTape() as tape:

            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient( cost_value, [X,W,b] )

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip(grads, [X,W,b]) )

        # Log periodically.
        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

    '''

    Training loss at iteration 0: 2567.1
    Training loss at iteration 20: 2349.2
    Training loss at iteration 40: 2203.4
    Training loss at iteration 60: 2102.9
    Training loss at iteration 80: 2031.8
    Training loss at iteration 100: 1980.0
    Training loss at iteration 120: 1941.5
    Training loss at iteration 140: 1912.0
    Training loss at iteration 160: 1889.1
    Training loss at iteration 180: 1870.9
    '''
    ```

Recommendations

- compute the ratings for all the movies and users and display the movies that are recommended
- these are based on the movies and ratings entered as my_ratings[] above
- to predict the rating of movie i for user j, compute w^j . x^i + b^j
- can be computed for all ratings using matrix multiplication

    ```python
    # Make a prediction using trained weights and biases
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

    #restore the mean
    pm = p + Ymean

    my_predictions = pm[:,0]

    # sort predictions
    ix = tf.argsort(my_predictions, direction='DESCENDING')

    for i in range(17):
        j = ix[i]
        if j not in my_rated:
            print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

    print('\n\nOriginal vs Predicted ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')

    '''
    Predicting rating 4.72 for movie Colourful (Karafuru) (2010)
    Predicting rating 4.65 for movie Odd Life of Timothy Green, The (2012)
    Predicting rating 4.62 for movie Deathgasm (2015)
    Predicting rating 4.60 for movie Eichmann (2007)
    Predicting rating 4.60 for movie Battle Royale 2: Requiem (Batoru rowaiaru II: Chinkonka) (2003)
    Predicting rating 4.60 for movie Into the Abyss (2011)
    Predicting rating 4.60 for movie Into the Forest of Fireflies' Light (2011)
    Predicting rating 4.60 for movie Raise Your Voice (2004)
    Predicting rating 4.59 for movie Particle Fever (2013)
    Predicting rating 4.58 for movie I'm the One That I Want (2000)

    Original vs Predicted ratings:

    Original 2.0, Predicted 2.18 for Shrek (2001)
    Original 5.0, Predicted 4.82 for Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    Original 1.0, Predicted 1.28 for Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)
    Original 5.0, Predicted 4.85 for Harry Potter and the Chamber of Secrets (2002)
    Original 5.0, Predicted 4.85 for Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    Original 5.0, Predicted 4.87 for Lord of the Rings: The Return of the King, The (2003)
    Original 3.0, Predicted 3.05 for Eternal Sunshine of the Spotless Mind (2004)
    Original 5.0, Predicted 4.88 for Incredibles, The (2004)
    Original 2.0, Predicted 2.11 for Persuasion (2007)
    Original 5.0, Predicted 4.72 for Toy Story 3 (2010)
    Original 5.0, Predicted 4.83 for Inception (2010)
    Original 1.0, Predicted 1.33 for Louis Theroux: Law & Disorder (2008)
    Original 1.0, Predicted 1.23 for Nothing to Declare (Rien à déclarer) (2010)
    '''
    ```

- in practice, additional information can be utilized to enhance the predictions
- the predicated ratings for the first few hundred movies lie in a small range
- you can augment by selecting from the top movies, movies that have high average ratings and movies with more than 20 ratings

    ```python
    filter=(movieList_df["number of ratings"] > 20)
    movieList_df["pred"] = my_predictions
    movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
    movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)
    ```