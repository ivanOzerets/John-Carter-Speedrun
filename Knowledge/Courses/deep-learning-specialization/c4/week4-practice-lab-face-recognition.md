- ideas presented from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) and [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)
- Face Verification: is this the claimed person
- Face Recognition: who is this person
- facenet learns a NN that encodes a face image into a vector of 128 numbers
- by comparing two such vectors, you can then determine if two pictures are of the same person

Packages

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL

%matplotlib inline
%load_ext autoreload
%autoreload 2
```

Naive Face Verification

- in face verification, given two images and have to determine if they are of the same person
- the simplest way to do this is to compare the two images pixel-by-pixel
- if the distance between the raw images is below a chosen threshold, it may be the same person

- this alg performs poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person’s face, minor changes in head position, ..
- rather than using the raw image, learn an encoding, f(img)
- by using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person

Encoding Face Images into a 128D Vector

- Using a ConvNet to Compute Encodings
    - the facenet model takes a lot of data and a long time to train
    - following the common practice in applied deep learning, load weights that someone else has already trained
    - the network architecture follows the inception model from [Szegedy *et al*..](https://arxiv.org/abs/1409.4842)
    - this network uses 160x160 dimensional rgb images as its input
    - specifically, a face image (or batch of m face images), as a tensor of shape (m, ny, nw, nc) = (m, 160, 160, 3)
    - the input images are originally of shape 96x96, thus, scale them to 160x160
    - this is done in the img_to_encoding() function
    - the output is a matrix of shape (m, 128) that encodes each input face image into a 128D vector

        ```python
        from tensorflow.keras.models import model_from_json

        json_file = open('keras-facenet-h5/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('keras-facenet-h5/model.h5')
        ```

        ```python
        print(model.inputs)
        print(model.outputs)

        '''
        [<tf.Tensor 'input_1:0' shape=(None, 160, 160, 3) dtype=float32>]
        [<tf.Tensor 'Bottleneck_BatchNorm/batchnorm/add_1:0' shape=(None, 128) dtype=float32>]
        '''
        ```

    - by using a 128 neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128
    - then use the encodings to compare two face images

    - so, an encoding is a good one if
        - the encodings of two images of the same person are quite similar to each other
        - the encodings of two images of different persons are very different
    - the triplet loss function formalizes this, and tries to push the encodings of two images of the same person (anchor and positive) closer together, while pulling the encodings of two images of different persons (anchor, negative) further apart

- The Triplet Loss
    - for an image x, its encoding is denoted as f(x), where f is the function computed by the NN

    - training will use triplets of images (A, P, N)
        - A is an anchor image - a picture of a person
        - P is a positive image - a picture of a different person than the anchor image
        - N is a negative image - a picture of a different person than the anchor image
    - these triplets are picked from the training dataset
    - (Ai, Pi, Ni) is used to denote the ith training example
    - want to make sure an image Ai of an individual is closer to the positive Pi than to the negative image Ni by at least a margin alpha

    - minimize the following triplet cost

    - here, the notation ‘[z]+’ is used to denote max(z, 0)
    - the term (1) is the squared distance between the anchor “A” and the positive “P” for a given triplet → want to be small
    - the term (2) is the squared distance between the anchor ‘A’ and the negative ‘N’ for a given triplet → want to be large
    - has a minus sign preceding because minimizing the negative of the term is the same as maximizing that term
    - alpha is called the margin, its a hyper that is picked manually, alpha = 0.2
    - most implementations also rescale the encoding vectors to have L2 norm equal to one
    - Exercise 1 - triple_loss
        - implement the triplet loss
            - compute the distance between the encodings of anchor and positive

            - compute the distance between the encodings of anchor and negative

            - compute the formula per training example

            - compute the full formula by taking the max with zero and summing over the training examples

        - useful functions: tf.reduce_sum(), tf.square(), tf.subtract(), tf.add(), tf.maximum()
        - for steps 1 and 2, sum over the entries of norms
        - for step 4, sum over the training examples
        - the square of the L2 norm is the sum of the squared differences

        - the anchor, positive and negative encodings are of shape (m, 128), where m is the number of training examples and 128 is the number of elements used to encode a single example
        - for steps 1 and 2, maintain the number of m training examples and sum along the 128 values of each encoding
        - tf.reduce_sum has an axis parameter, this chooses along which axis the sums are applied
        - one way to choose the last axis in a tensor is to use negative indexing (axis = -1)
        - when summing over training examples, the result will be a single scalar value
        - for tf.reduce_sum to sum across all axes, keep the default value axis=None

            ```python
            # UNQ_C1(UNIQUE CELL IDENTIFIER, DO NOT EDIT)
            # GRADED FUNCTION: triplet_loss

            def triplet_loss(y_true, y_pred, alpha = 0.2):
                """
                Implementation of the triplet loss as defined by formula (3)

                Arguments:
                y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
                y_pred -- python list containing three objects:
                        anchor -- the encodings for the anchor images, of shape (None, 128)
                        positive -- the encodings for the positive images, of shape (None, 128)
                        negative -- the encodings for the negative images, of shape (None, 128)

                Returns:
                loss -- real number, value of the loss
                """

                anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

                ### START CODE HERE
                #(≈ 4 lines)
                # Step 1: Compute the (encoding) distance between the anchor and the positive
                pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
                # Step 2: Compute the (encoding) distance between the anchor and the negative
                neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
                # Step 3: subtract the two previous distances and add alpha.
                basic_loss = pos_dist - neg_dist + alpha
                # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
                loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
                ### END CODE HERE

                return loss

            # loss = tf.Tensor(527.2598, shape=(), dtype=float32)
            ```

Loading the Pre-trained Model

- FaceNet is trained by minimizing the triplet loss
- but since training requires a lot of data and a lot of computation, take weights from pre-trained model

    ```python
    FRmodel = model
    ```

- some examples of distances between the encodings between three individuals

Applying the Model

- building a system for an office building where the building manager would like to offer facial recognition to allow the employees to enter the building
- build a face verification system that gives access to a list of people
- to be admitted, each person has to swipe an identification card at the entrance
- the face recognition system then verifies that they are who they claim to be
- Face Verification
    - build a database containing one encoding vector for each person who is allowed to enter the office
    - to generate the encoding, use img_to_encoding(image_path, model), which runs the forward prop of the model on the specified image
    - build the database (represented as a python dictionary)
    - this database maps each person’s name to a 128 encoding of their face

        ```python
        #tf.keras.backend.set_image_data_format('channels_last')
        def img_to_encoding(image_path, model):
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
            img = np.around(np.array(img) / 255.0, decimals=12)
            x_train = np.expand_dims(img, axis=0)
            embedding = model.predict_on_batch(x_train)
            return embedding / np.linalg.norm(embedding, ord=2)
        ```

        ```python
        database = {}
        database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
        database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
        database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
        database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
        database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
        database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
        database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
        database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
        database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
        database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
        database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
        database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
        ```

    - load images of some faces

        ```python
        danielle = tf.keras.preprocessing.image.load_img("images/danielle.png", target_size=(160, 160))
        kian = tf.keras.preprocessing.image.load_img("images/kian.jpg", target_size=(160, 160))
        ```

        ```python
        np.around(np.array(kian) / 255.0, decimals=12).shape

        # (160, 160, 3)
        ```

        ```python
        kian
        ```

        ```python
        np.around(np.array(danielle) / 255.0, decimals=12).shape

        # (160, 160, 3)
        ```

        ```python
        danielle
        ```

    - now, when someone shows up at the front door and spies their id card (providing a name), look up their encoding in the database, and use it to check if the person standing at the front door matches the name on the ID
    - Exercise 2 - verify
        - implement the verify() function, which checks if the front-door camera picture (image_path) is actually the person called identity
            - compute the encoding of the image from image_path
            - compute the distance between this encoding and the encoding of the identity image stored in the database
            - open the door if the distance is less than 0.5, else do not open
        - use the L2 distance np.linalg.norm
        - compare the L2 distance, not the square of the L2 distance, to the threshold 0.7
        - identity is a string that is also a key in the database dictionary
        - img_to_encoding has two parameters: the image_path and model

        ```python
        # UNQ_C2(UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: verify

        def verify(image_path, identity, database, model):
            """
            Function that verifies if the person on the "image_path" image is "identity".

            Arguments:
                image_path -- path to an image
                identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
                database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
                model -- your Inception model instance in Keras

            Returns:
                dist -- distance between the image_path and the image of "identity" in the database.
                door_open -- True, if the door should open. False otherwise.
            """
            ### START CODE HERE
            # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
            encoding = img_to_encoding(image_path, model)
            # Step 2: Compute distance with identity's image (≈ 1 line)
            dist = np.linalg.norm(encoding - database[identity])
            # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
            if dist < 0.7:
                print("It's " + str(identity) + ", welcome in!")
                door_open = True
            else:
                print("It's not " + str(identity) + ", please go away")
                door_open = False
            ### END CODE HERE        
            return dist, door_open
        ```

    - run the verification alg

        ```python
        # BEGIN UNIT TEST
        distance, door_open_flag = verify("images/camera_0.jpg", "younes", database, FRmodel)
        assert np.isclose(distance, 0.5992949), "Distance not as expected"
        assert isinstance(door_open_flag, bool), "Door open flag should be a boolean"
        print("(", distance, ",", door_open_flag, ")")
        # END UNIT TEST

        '''
        It's younes, welcome in!
        ( 0.5992949 , True )
        '''
        ```

        ```python
        verify("images/camera_2.jpg", "kian", database, FRmodel)

        '''
        It's not kian, please go away

        (1.0259346, False)
        '''
        ```

- Face Recognition
    - change the face verification system to a face recognition system
    - this way, no one has to carry an id card anymore
    - implement a face recognition system that takes as input an image, and figures out if it is one of the authorized persons
    - unlike the previous face verification system, no longer get a person’s name
    - Exercise 3 - who_is_it
        - compute the target encoding of the image from image_path
        - find the encoding from the database that has smallest distance with the target encoding
        - initialize the min_dist variable to a large enough number
        - this helps to keep track of the closest encoding to the input’s encoding
        - loop over the database dictionary’s names and encodings
        - to loop use for (name, db_enc) in database.items()
        - compute the L2 distance between the target ‘encoding’ and the current ‘encoding’ from the database
        - if the distance is less than the min_dist, then set min_dist to dist, and identity to name

        ```python
        # UNQ_C3(UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: who_is_it

        def who_is_it(image_path, database, model):
            """
            Implements face recognition for the office by finding who is the person on the image_path image.

            Arguments:
                image_path -- path to an image
                database -- database containing image encodings along with the name of the person on the image
                model -- your Inception model instance in Keras

            Returns:
                min_dist -- the minimum distance between image_path encoding and the encodings from the database
                identity -- string, the name prediction for the person on image_path
            """

            ### START CODE HERE

            ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
            encoding =  img_to_encoding(image_path, model)

            ## Step 2: Find the closest encoding ##

            # Initialize "min_dist" to a large value, say 100 (≈1 line)
            min_dist = 100

            # Loop over the database dictionary's names and encodings.
            for (name, db_enc) in database.items():

                # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
                dist = np.linalg.norm(encoding - db_enc)

                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
                if dist < min_dist:
                    min_dist = dist
                    identity = name
            ### END CODE HERE

            if min_dist > 0.7:
                print("Not in the database.")
            else:
                print ("it's " + str(identity) + ", the distance is " + str(min_dist))

            return min_dist, identity
        ```

        ```python
        ### YOU CANNOT EDIT THIS CELL

        # BEGIN UNIT TEST
        # Test 1 with Younes pictures 
        who_is_it("images/camera_0.jpg", database, FRmodel)

        # Test 2 with Younes pictures 
        test1 = who_is_it("images/camera_0.jpg", database, FRmodel)
        assert np.isclose(test1[0], 0.5992946)
        assert test1[1] == 'younes'

        # Test 3 with Younes pictures 
        test2 = who_is_it("images/younes.jpg", database, FRmodel)
        assert np.isclose(test2[0], 0.0)
        assert test2[1] == 'younes'
        # END UNIT TEST

        '''
        it's younes, the distance is 0.5992949
        it's younes, the distance is 0.5992949
        it's younes, the distance is 0.0
        '''
        ```