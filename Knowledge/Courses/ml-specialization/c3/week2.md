Making recommendations

- recommender systems
- Predicting movie ratings
    - rating zero to five stars
    - users and items
    - n_u = no. of users
    - n_m = no. of movies
    - r(i,j) = 1 if user j has rated move i
    - y^(i,j) = rating given by user j to movie i (defined only if r(i,j) = 1)
    - look at movies users have not rated and try to predict how users will rate those movies because then its more likely to recommend to users movies that they will rate five stars
    - assume features are know about the movies

Using per-item features

- What if we have features of the movies?
    - n is the number of features
    - for user 1: predict rating for movie i as: w . x^i + b → just linear regression

    - For user j: predict user j’s rating for movie i as w^j . x^i + b^j
    - different linear regression model for each of the users
- Cost function
    - r(i,j) = 1 → if user j has rated movie i (0 otherwise)
    - y^(i,j) = rating given by user j on movie i (if defined)
    - w^j, b^j = parameters for user j
    - x^i = feature vector for movie i
    - For user j and movie i, predict rating: w^j . x^i + b^j
    - m^j = no. of movies rated by user j
    - to learn w^j, b^j, for one user

    - can eliminate m^j constant, should end up with the same value of w and b
    - to learn parameters w^1, b^1, …, w^(n_u),b^(n_u) for all users:

Collaborative filtering alg

- Problem motivation
    - if no features for the movies are available
    - if you have the parameters and rating to match, you can take a reasonable guess
    - multiple ratings for the same movie allow for collaborative filtering
    - so multiple linear regression models for the same movie can be used to guess the initial features
- Cost function
    - given w^i, b^i, …, w^(n_u), b^(n_u)
    - to  learn x^i

    - to learn x^i, …, x^(n_m)

- Collaborative filtering
    - combine cost functions since double sum is the same in both functions

- Gradient Descent
    - same as linear regression except with x

Binary labels: favs, likes and clicks

- Binary labels
- Example application
    - did user j purchase an item after being shown
    - did user j fav/like an item
    - did user j spend at least 30 sec with an item
    - did user j click on an item
- From regression to binary classification
    - previously:
        - predict y^(i,j) as w^j . x^i + b^j
    - for binary labels:
        - predict that the probability of y^(i,j) = 1 is given by g(w^j . x^i + b^i), where

- Cost function for binary application
    - previous cost function

    - loss for binary labels

    Refer to the table above for question 1 and 2. Assume numbering starts at 1 for this quiz, so the rating for Football Forever by Elissa is at (1,1). What is the value of n_u? → 4

- What is the value of r(2,2) → 0
- In which of the following situations will a collaborative filtering system be the most appropriate learning algorithm (compared to linear or logistic regression)?
    - [ ]  you manage an online bookstore and you have the book ratings from many users. You want to learn to predict the expected sales volume (number of books sold) as a function of the average rating of a book
    - [x]  you run an online bookstore and collect the ratings of many users. You want to use this to identify what books are "similar" to each other (i.e., if a user likes a certain book, what are other books that they might also like?)
    - [ ]  you subscribe to an online video streaming service, and are not satisfied with their movie suggestions. You download all your viewing for the last 10 years and rate each item. You assign each item a genre. Using your ratings and genre assignment, you learn to predict how you will rate new movies based on the genre
    - [ ]  you're an artist and hand-paint portraits for your clients. Each client gets a different portrait (of themselves) and gives you 1-5 star rating feedback, and each client purchases at most 1 portrait. You'd like to predict what rating your next customer will give you
- For recommender systems with binary labels y, which of these are reasonable ways for defining when y should be 1 for a given user j and item i?
    - [ ]  y is 1 if user j has been shown item i by the recommendation engine
    - [ ]  u is 1 if user j has not yet been shown item i by the recommendation engine
    - [x]  y is 1 if user j purchases item i (after being shown the item)
    - [x]  y is 1 if user j fav/likes/clicks on item i (after being shown the item)

Mean normalization

- algorithm currently predicts new users will rate all movies with zero stars, not helpful
- take ratings in matrix and take averages

- for user j, on movie i predict: x^j . x^i + b^j + mu_i

- new users will be initialized to the average rating
- normalizing columns would help for a brand new movie

TensorFlow implementation of collaborative filtering

- Derivatives in ML
    - turns out, computing the partial derivative term in the update for GD can be difficult
- Custom Training Loop
    - if fixed cost function J = (wx - 1)^2
    - Tf.variables are the parameters to optimize

    - known as Auto Diff

Finding related items

- Limitations of collaborative filtering
    - Cold start problem, how to
        - rank new items that few users have rated
        - show something reasonable to new users who have rated few items
    - Use side information about items or users:
        - Item: genre, movie starts, studio, …
        - User: demographics, expressed preferences, …

Implementation in TensorFlow

- Finding related items
    - the features x^i of item i are quite hard to interpret
    - the learned features n collectively convey something about what that movie is like
    - to find other items related to it, find item k with x^k similar to x^i

- J_train(w,b) is the fraction of the train set that has been misclassified

- the reason you cant use a Dense layer and model.compile() and model.fit() is because the collaborative filtering alg and its cost function doesn’t uniquely fit into standard NN layer types

    Using the table below, find the closest item to the movie “Pies, Pies, Pies”. 

    - [ ]  Pastries for Supper
    - [ ]  Pies for You
- Which of these is an example of the cold start problem?
    - [x]  a recommendation system is unable to give accurate rating predictions for a new product that no users have rated
    - [ ]  a recommendation system is so computationally expensive that it causes your computer CPU to heat up, causing your computer to need to be cooled down and restarted
    - [x]  a recommendation system is unable to give accurate rating predication for a new user that has rated few products
    - [ ]  a recommendation takes so long to train that users get bored and leave

Collaborative filtering vs content-based filtering

- Collaborative filtering vs content-based filtering
    - collaborative filtering
        - recommend items to you based on rating of users who gave similar ratings as you
    - content-based filtering
        - recommend items to you based on features of user and item to find a good match
    - r(i,j = 1 if user j has rated item i
    - y^(i,j) rating given by user j on item i
    - key is to make good use of features of users and items
- Examples of user and item features
    - user features → x_u^j for user j
        - age
        - gender
        - country
        - movies watched
        - average rating per genre
    - movie features → x_m^i for movie i
        - year
        - genre/genres
        - reviews
        - average rating
- Content-based filtering: Learning to match
    - predict rating of user j on movie i as
    - v_u^j → computed from x_u^j
    - v_m^i → computed from x_m^i

Deep learning for content-based filtering

- NN architecture
    - user network

    - movie network

    - predication: v_u . v_m
    - output layers must have same dimension
    - g(v_u . v_m) to predict the probability that y^(i,j) is 1
    - can be drawn as a single network

    - assuming there are some data of some user having rated some movies

- Learned user and item vectors
    - v_u^j is a vector of length 32 that describes user j with features x_u^j
    - v_m^i is a vector of length 32 that describes user j with features x_m^i
    - to find movies similar to movie i

Recommending from a large catalogue

- How to efficiently find recommendation from a large set of items
    - movies 1000+
    - ads 1m+
    - songs 10m+
    - products 10m+
- Two steps: retrieval and ranking
    - retrieval
        - generate large list of plausible item candidates
            - for each of the last 10 movies watched by the user, find 10 most similar movies
            - for most viewed 3 genres, find the top 10 movies
            - can be precomputed as a good starting point

            - for most viewed 3 genres, find the top 10 movies
            - top 20 movies in the country
        - combine retrieved items into list, removing duplicates and items already watched/purchased
        - goal to have broad coverage
    - ranking
        - take list retrieved and rank using learned model
        - display ranked items to user
        - run through NN with the smaller list
        - if v_m is computed already, just run with new x_u
- Retrieval step
    - retrieving more items results in better performance, but slower recommendations
    - to analyse/optimize the trade-off, carry out offline experiments to see if retrieving additional items results in more relevant recommendations (p(y(i,j) = 1 of items displayed to user are higher)

Ethical use of recommender systems

- What is the goal of the recommender system?
    - recommend
        - movies most likely to be rated five stars by user
        - products most likely to be purchased
        - ads most likely to be clicked on
        - products generating the largest profit
        - video leading to maximum watch times
- Ethical considerations with recommender systems
    - travel industry
        - good travel experience to more users

    - payday loans
        - squeeze customers more

    - amelioration: do not accept ads from exploitative businesses
- Other problematic cases
    - maximizing user engagement has led to large social media/video sharing sites to amplify conspiracy theories and hate/toxicity
    - amelioration: filter out problematic content such as hate speech, fraud, scams, and violent content
    - can a ranking system maximize your profit rather than users’ welfare be presented in a transparent way
    - amelioration: be transparent with users

TensorFlow implementation of content-based filtering

- normal NN
- normalize vector vu to be length 1
- special layer for dot product

- normalizing the l2 norm of the vector makes the alg work better

Reducing the number of features

- PCA lets you take data with many features and reduce it to two or three so you can plot it and visualize it
- Car measurements
    - if you want to reduce the number of features, take features that vary the most?
    - find one or more new axis and coordinates
    - use fewer numbers to capture ‘size’ feature

- Data visualization
    - reduce 50 features to two

PCA algorithm

- PCA alg
    - unsupervised
    - preprocess features
    - normalized to have zero mean
    - feature scaling so ranges are not too far apart

- Choose an axis
    - project examples onto the new axis

    - variance is large
    - capturing info of original data

    - less variance → capturing much less of the information

    - principal component has max variance
- Coordinate on the new axis
    - project on axis using the dot product

- More principal components
    - 2nd principal component and subsequent principle component are perpendicular to each other
- PCA is not linear regression
    - linear regression ties to fit a straight so that the predicted value is as close as possible to the ground truth label y, in the vertical direction
    - PCA → not trying to fit a line, no ground truth label, treats x1 and x2 equally and tries to find a line that will make closest distance the smallest
    - linear regression → y is treated as very special and minimize line segments in the y direction
    - find axis to retain variance (info)

- Approximation to the original data
    - reconstruction approximates the original length x length 1 vector

PCA in code

- PCA in scikit-learn
    1. optional pre-processing: perform feature scaling

        - fit the data to obtain 2 (or 3) new axes (principal components)
        - fit() function in scikit-learn automatically carries out mean normalization

    2. optionally examine how much variance is explained by each principal component
        - explained_variance_ratio()
    3. transform (project) the data onto the new axes
        - transform()
- Example

- Application of PCA
    - Visualization to reduce to 2 or 3 features
    - Less frequently used for
        - data compression (to reduce storage or transmission costs)
        - speed up training of a supervised learning model