Implementing K-means

- clusters similar data points together
- given training set x^1, …, x^m
- looking to group the data into a few cohesive ‘clusters’
- k-means is an iterative procedure that
    - starts by guessing the initial centroids, and then
    - refines the guess by
        - repeatedly assigning examples to their closest centroids
        - recomputing the centroids based on the assignments
    - in pseudocode, the k-means alg

    - inner-loop of the alg repeatedly carries out two steps
        - assign each training example x^i to its closest centroid
        - recompute the mean of each centroid using the points assigned to it
    - the K-means alg will always converge to some final set of means for the centroids
    - however, the converged solution may not always be ideal and depends on the initial setting of the centroids
        - therefore, in practice the K-means alg is usually run a few times with different random initializations
        - one way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion)
- implement the two phases of the K-means alg separately
    - start by completing find_closest_centroid and then complete compute_centroids
- Finding closest centroids
    - the alg assigns every training example x^i to its closest centroid, given the current positions of centroids
    - Exercise 1
        - complete the code in find_closest_centroids
        - this function takes the data matrix X and the locations of all centroids
        - should output a 1D array idx (which has the same number of elements as X) that holds the index of the closest centroid (a value in {1, …, K-1}, where K is total number of centroids) to every training example
        - specifically, for every example x^i we set

            - c^i is the index of the centroid that is closest to x^i (corresponds to idx[i])
            - mui_j is the position (value) of the jth centroid
            - L2-norm is the abs of x^i - mui_j

            ```python
            # UNQ_C1
            # GRADED FUNCTION: find_closest_centroids

            def find_closest_centroids(X, centroids):
                """
                Computes the centroid memberships for every example

                Args:
                    X (ndarray): (m, n) Input values      
                    centroids (ndarray): (K, n) centroids

                Returns:
                    idx (array_like): (m,) closest centroids

                """

                # Set K
                K = centroids.shape[0]

                # You need to return the following variables correctly
                idx = np.zeros(X.shape[0], dtype=int)

                ### START CODE HERE ###

                m = X.shape[0]

                for i in range (m):
                    norms = []
                    for j in range (K):
                        l2 = np.linalg.norm(X[i]-centroids[j])
                        norms.append(l2)
                    ci = np.argmin(norms)
                    idx[i] = ci

                 ### END CODE HERE ###

                return idx
            ```

            ```python
            # Load an example dataset that we will be using
            X = load_data()
            ```

            ```python
            print("First five elements of X are:\n", X[:5]) 
            print('The shape of X is:', X.shape)

            '''
            First five elements of X are:
             [[1.84207953 4.6075716 ]
             [5.65858312 4.79996405]
             [6.35257892 3.2908545 ]
             [2.90401653 4.61220411]
             [3.23197916 4.93989405]]
            The shape of X is: (300, 2)
            '''
            ```

            ```python
            # Select an initial set of centroids (3 Centroids)
            initial_centroids = np.array([[3,3], [6,2], [8,5]])

            # Find closest centroids using initial_centroids
            idx = find_closest_centroids(X, initial_centroids)

            # Print closest centroids for the first three elements
            print("First three elements in idx are:", idx[:3])

            # UNIT TEST
            from public_tests import *

            find_closest_centroids_test(find_closest_centroids)

            '''
            First three elements in idx are: [0 2 1]
            All tests passed!
            '''
            ```

- Computing centroid means
    - given assignments of every point to a centroid, the second phase of the alg recomputes, for each centroid, the mean of the points that were assigned to it
    - Exercise 2
        - complete the compute_centroids below to recompute the value for each centroid
            - for every centroid mui_k, set

            - c_k is the set of examples that are assigned to centroid k
            - abs(c_k) is the number of examples in the set C_k
        - if two examples say x^3 and x^5 are assigned to centroid k = 2, then you should update mui_2 = (x^3 + x^5)/2

            ```python
            # UNQ_C2
            # GRADED FUNCTION: compute_centroids

            def compute_centroids(X, idx, K):
                """
                Returns the new centroids by computing the means of the 
                data points assigned to each centroid.

                Args:
                    X (ndarray):   (m, n) Data points
                    idx (ndarray): (m,) Array containing index of closest centroid for each 
                                   example in X. Concretely, idx[i] contains the index of 
                                   the centroid closest to example i
                    K (int):       number of centroids

                Returns:
                    centroids (ndarray): (K, n) New centroids computed
                """

                # Useful variables
                m, n = X.shape

                # You need to return the following variables correctly
                centroids = np.zeros((K, n))

                ### START CODE HERE ###

                for k in range(K):
                    points = X[idx == k]
                    centroids[k] = np.mean(points, axis=0)

                ### END CODE HERE ## 

                return centroids
            ```

            ```python
            K = 3
            centroids = compute_centroids(X, idx, K)

            print("The centroids are:", centroids)

            # UNIT TEST
            compute_centroids_test(compute_centroids)

            '''
            The centroids are: [[2.42830111 3.15792418]
             [5.81350331 2.63365645]
             [7.11938687 3.6166844 ]]
            All tests passed!
            '''
            ```

K-means on a sample dataset

- the next step is to run the K-means alg on a toy 2D dataset
- the code below will produce a visualization that steps through the progress of the alg at each iteration

    ```python
    # You do not need to implement anything for this part

    def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
        """
        Runs the K-Means algorithm on data matrix X, where each row of X
        is a single example
        """

        # Initialize values
        m, n = X.shape
        K = initial_centroids.shape[0]
        centroids = initial_centroids
        previous_centroids = centroids    
        idx = np.zeros(m)
        plt.figure(figsize=(8, 6))

        # Run K-Means
        for i in range(max_iters):

            #Output progress
            print("K-Means iteration %d/%d" % (i, max_iters-1))

            # For each example in X, assign it to the closest centroid
            idx = find_closest_centroids(X, centroids)

            # Optionally plot progress
            if plot_progress:
                plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
                previous_centroids = centroids

            # Given the memberships, compute new centroids
            centroids = compute_centroids(X, idx, K)
        plt.show() 
        return centroids, idx
    ```

    ```python
    # Load an example dataset
    X = load_data()

    # Set initial centroids
    initial_centroids = np.array([[3,3],[6,2],[8,5]])

    # Number of iterations
    max_iters = 10

    # Run K-Means
    centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
    ```

Random initialization

- a good strategy for initializing the centroids is to select random examples from the training set
- understand the function kMeans_init_centroids
    - first randomly shuffle the indices of the examples (np.random.permutation())
    - select the first K examples based on the random permutation of the indices
    - allow the examples to be selected at random without the risk of selecting the same example twice

    ```python
    # You do not need to modify this part

    def kMeans_init_centroids(X, K):
        """
        This function initializes K centroids that are to be 
        used in K-Means on the dataset X

        Args:
            X (ndarray): Data points 
            K (int):     number of centroids/clusters

        Returns:
            centroids (ndarray): Initialized centroids
        """

        # Randomly reorder the indices of examples
        randidx = np.random.permutation(X.shape[0])

        # Take the first K examples as centroids
        centroids = X[randidx[:K]]

        return centroids
    ```

    ```python
    # Run this cell repeatedly to see different outcomes.

    # Set number of centroids and max number of iterations
    K = 3
    max_iters = 10

    # Set initial centroids by picking random examples from the dataset
    initial_centroids = kMeans_init_centroids(X, K)

    # Run K-Means
    centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
    ```

Image compression with K-means

- apply K-means to image compression
    - in a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values
    - this encoding is often referred to as the RGB encoding
    - the image contains thousands of colors, reduce the number of colors to 16 in this exercise
    - by making this reduction, it is possible to represent (compress) the photo in an efficient way
    - specifically, store the RGB values of the 16 selected colors, and for each pixel in the image you now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities)
- use k-means to select the 16 colors that will be used to represent the compressed image
    - concretely, treat every pixel in the original image as a data example and use the k-menas alg to find the 16 colors that best group (cluster) the pixels in the 3D RGB space
    - once you have computed the cluster centroids on the image, use the 16 colors to replace the pixels in the original image

- Dataset
    - Load image
        - use matplotlib to read in the original image

        ```python
        # Load an image of a bird
        original_img = plt.imread('bird_small.png')
        ```

    - Visualize image
        - visualize the image that was just loaded

        ```python
        # Visualizing the image
        plt.imshow(original_img)
        ```

    - Check the dimension of the variable
        - print out the shape of the variable to get more familiar with the data

        ```python
        print("Shape of original_img is:", original_img.shape)

        # Shape of original_img is: (128, 128, 3)
        ```

        - this creates a 3D matrix original_img
            - the first two indices identify a pixel position
            - the third index represents red, green, or blue
        - for example, originial_img[50,33,2] gives the blue intensity of the pixel at row 50 and column 33
    - Processing data
        - call run_kMeans, first transform the matrix original_img into a 2D matrix
            - reshape the matrix original_img to create an m x 3 matrix of pixel colors (where m = 16384 = 128 x 1128)
        - Note: if you use a JPG file, first divide the pixel values by 255 so it will be in the range 0 to 1

        ```python
        # Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
        # original_img = original_img / 255

        # Reshape the image into an m x 3 matrix where m = number of pixels
        # (in this case m = 128 x 128 = 16384)
        # Each row will contain the Red, Green and Blue pixel values
        # This gives us our dataset matrix X_img that we will use K-Means on.

        X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
        ```

- K-Means on image pixels
    - run K-means on the pre-processed image

        ```python
        # Run your K-Means algorithm on this data
        # You should try different values of K and max_iters here
        K = 16
        max_iters = 20

        # Using the function you have implemented above. 
        initial_centroids = kMeans_init_centroids(X_img, K)

        # Run K-Means - this can take a couple of minutes depending on K and max_iters
        centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
        ```

        ```python
        print("Shape of idx:", idx.shape)
        print("Closest centroid for the first five elements:", idx[:5])

        '''
        Shape of idx: (16384,)
        Closest centroid for the first five elements: [ 2 10 10  2  2]
        '''
        ```

    - the code below will plot all the colors found int he original image
    - the color of each pixel is represented by RGB values so the plot should have 3 axis - R, G, B
    - notice a lot of dots below representing thousands of colors in the original image
    - the red markers represent the centroids after running K-means
    - these will be the 16 colors that will be used to compress the image

        ```python
        # Plot the colors of the image and mark the centroids
        plot_kMeans_RGB(X_img, centroids, idx, K)
        ```

    - visualize the colors at each of the red markers (the centroids) above with the function below
    - the number below each color is its index and these numbers are the numbers in the idx array

        ```python
        # Visualize the 16 colors selected
        show_centroid_colors(centroids)
        ```

- Compress the image
    - find the top K = 16 colors to represent the image
    - now assign each pixel position to its closest centroid using the find_closest_centroids function
        - allows to represent the original image using the centroid assignments of each pixel
        - notice the significantly reduced number of bits that are required to describe the image
            - the original image required 24 bits for each one of the 128 x 128 pixel locations, resulting in total size of 128 x 128 x 24 = 393,216 bits
            - the new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location
            - the final number of bits used is therefore 16 x 24 + 128 x 128 x 4 = 65,920 bits, which corresponds to compressing the original image by about a factor of 6

        ```python
        # Find the closest centroid of each pixel
        idx = find_closest_centroids(X_img, centroids)

        # Replace each pixel with the color of the closest centroid
        X_recovered = centroids[idx, :] 

        # Reshape image into proper dimensions
        X_recovered = np.reshape(X_recovered, original_img.shape) 
        ```

    - view the effects of the compression by reconstructing the image based only on the centroid assignments
        - replaced each pixel with the value of the centroid assigned to it
        - even though the resulting image retains most of the characteristics of the original, some compression artifacts can be seen

        ```python
        # Display original image
        fig, ax = plt.subplots(1,2, figsize=(16,16))
        plt.axis('off')

        ax[0].imshow(original_img)
        ax[0].set_title('Original')
        ax[0].set_axis_off()

        # Display compressed image
        ax[1].imshow(X_recovered)
        ax[1].set_title('Compressed with %d colours'%K)
        ax[1].set_axis_off()
        ```