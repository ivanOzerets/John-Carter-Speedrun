Packages

```python
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import pprint
from public_tests import *
%matplotlib inline
```

Problem Statement

- Neural Style transfer (NST) is one of the most fun and interesting optimization techniques in deep learning
- it merges two images, namely: a content image (C) and a style image (S), to create a generated image (G)
- the generated image G combines the content of the image C with the style of image S
- combine the Louvre museum in Paris with the impressionist style of Claude Monet

Transfer Learning

- NST uses a previously trained conv network, and build on top
- the idea of using a network trained on a different task and applying it to a new task is called transfer learning
- using the eponymously named VGG network from the [original NST paper](https://arxiv.org/abs/1508.06576) published by the Visual Geometry Group at University of Oxford
- specifically, the VGG-19, a 19 layer version of the VGG network
- this model has already been trained on the very large imagenet database, and has learned to recognize a variety of low and high level features

```python
tf.random.set_seed(272) # DO NOT CHANGE THIS VALUE
pp = pprint.PrettyPrinter(indent=4)
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg.trainable = False
pp.pprint(vgg)
```

Neural Style Transfer (NST)

- first, build the content cost function Jcontent(C, G)
- second, build the style cost function Jstyle(S, G)
- finally, put together to get J(G) = alphaJcontent(C, G) + betaJstyle(S, G)
- Computing the Content cost
    - Make Generated Image G Match the Content of Image C
        - goal when performing NST is for the content in generated image G to match the content of image C
            - the shallower layers of a ConvNet tent to detect lower-level features such as edges and simple textures
            - the deeper layer tend to detect high-level features such as more complex textures and object classes
        - to choose a middle activation layer a[l]
            - need the generated image G to have similar content as the input image C
            - choose some layer’s activations to represent the content of an image
            - in practice, usually get the most visually pleasing results when choosing a layer from somewhere in the middle of the network
            - this ensures that the network detects both higher-level and lower-level features
        - to forward prop image C
            - set the image C as the input to the pretrained VGG network, and run forward prop
            - let a(C) be the hidden layer activations in the chosen layer
            - this will be an nh x nw x nc tensor
        - to forward prop image G
            - repeat the process with the image G
            - set G as the input and run forward prop
            - let a(G) be the corresponding hidden layer activation
        - the content image C will be the picture of the Louvre Museum in Paris

            ```python
            content_image = Image.open("images/louvre.jpg")
            print("The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.")
            content_image
            ```

    - Content Cost Function Jcontent(C, G)
        - one goal when performing NST is for the content in generated image G to match the content of image C
        - a method to achieve this is to calculate the content cost function

        - here nh, nw, and nc are the height, width and number of channels of the hidden layer chosen, and appear in a normalization term in the cost
        - for clarity, note that a(C) and a(G) are the 3D volumes corresponding to a hidden layer’s activations
        - in order to compute the cost Jcontent(C, G), it might also be convenient to unroll the 3D volumes into a 2D matrix
        - technically this unrolling step isn’t needed to compute Jcontent, but it will be good practice for when a similar operation need to be carried out later for computing the style cost Jstyle

        - Exercise 1 - compute_content_cost
            - a_G: hidden layer activations representing content of the image G
            - a_C: hidden layer activations representing content of the image C
            - retrieve dimensions from a_G
                - to retrieve dimensions from a tensor X, use: X.get_shape().as_list()
            - unroll a_C and a_G
                - use tf.transpose and tf.reshape
            - compute the content cost
                - use tf.reduce_sum, tf.square, tf.subtract
            - to unroll the tensor, want the shape to change from (m, nh, nw, nbc) to (m, nh x nw, nc)
            - tf.reshpae(tensor, shape) takes a list of integers that represent the desired output shape
            - for the shape parameter, a -1 tells the function to choose the correct dimension size so that the output tensor still contains all the values of the original tensor
            - so tf.reshape(a_C, shape=[m, n_H * n_W, n_C]) gives the same result as tf.reshape(a_C, shape=[m, -1, n_C])
            - to re-order the dimensions, you can use tf.transpose(tensor, perm), where perm is a list of integers containing the original index of the dimensions
            - for example, tf.transpose(a_C, perm=[0,3,1,2] changes the dimensions from (m, nh, nw, nc) to (m, nc, nh, nw)
            - don’t necessarily need tf.transpose to unroll the tensors in the case but this is a useful function to practice and understand for other situations

            ```python
            # UNQ_C1
            # GRADED FUNCTION: compute_content_cost

            def compute_content_cost(content_output, generated_output):
                """
                Computes the content cost

                Arguments:
                a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
                a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

                Returns: 
                J_content -- scalar that you compute using equation 1 above.
                """
                a_C = content_output[-1]
                a_G = generated_output[-1]

                ### START CODE HERE

                # Retrieve dimensions from a_G (≈1 line)
                _, n_H, n_W, n_C = a_G.shape

                # Reshape 'a_C' and 'a_G' (≈2 lines)
                # DO NOT reshape 'content_output' or 'generated_output'
                a_C_unrolled = tf.reshape(a_C, shape=[1,-1,n_C])
                a_G_unrolled = tf.reshape(a_G, shape=[1,-1,n_C])

                # compute the cost with tensorflow (≈1 line)
                J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

                ### END CODE HERE

                return J_content
            ```

- Computing the Style Cost
    - use the following style image

        ```python
        example = Image.open("images/monet_800600.jpg")
        example
        ```

    - painted in the style of impressionism
    - Style Matrix
        - Gram matrix
            - the style matrix is also called a Gram matrix
            - in linear algebra, the gram matrix G of a set of vectors (v1, …, vn) is the matrix of dot products, whose entries are Gij = viTvj = np.dot(vi, vj)
            - Gij compares how similar vi is to vj
            - if they are highly similar, expect them to have a large dot product, and thus for Gij to be large
        - Two meaning of the variable G
            - there is an unfortunate collision in the variable names
            - following the common terminology used
                - G is used to denote the Style matrix
                - G also denotes the generated image
            - Ggram will refer to the Gram matrix, and G will denote the generated image
        - Compute Gram matrix Ggram
            - compute the style matrix by multiplying the unrolled filter matrix with its transpose

        - G(gram)ij: correlation
            - the result is a matrix (nc, nc) where nc is the number of filters (channels)
            - the value G(gram)ij measures how similar the activations of filter i are to the activation of filter j
        - G(gram), ii: prevalence of patterns or textures
            - the diagonal elements G(gram)ii measure how active a filer i is
            - suppose filter i is detecting vertical textures in the image, then G(gram)ii measures how common vertical textures are in the image as a whole
            - if G(gram)ii is large, this means that the image has a lot of vertical texture
        - by capturing the prevalence of different types of features (G(gram)ii), as well as how much different features occur together (G(gram)ij), the Style matrix Ggram measures the style of an image
        - Exercise 2 - gram_matrix
            - using tf, implement a function that compute the gram matrix of a matrix A
            - GA = AAT
            - use matmul and transpose

                ```python
                # UNQ_C2
                # GRADED FUNCTION: gram_matrix

                def gram_matrix(A):
                    """
                    Argument:
                    A -- matrix of shape (n_C, n_H*n_W)

                    Returns:
                    GA -- Gram matrix of A, of shape (n_C, n_C)
                    """  
                    ### START CODE HERE

                    #(≈1 line)
                    GA = A @ np.transpose(A)

                    ### END CODE HERE

                    return GA

                '''
                GA = 
                tf.Tensor(
                [[ 63.193256  -26.729713   -7.732155 ]
                 [-26.729713   12.775055   -2.5164719]
                 [ -7.732155   -2.5164719  23.746586 ]], shape=(3, 3), dtype=float32)
                '''
                ```

        - Style Cost
            - minimize the distance between the gram matrix of the style image S and the gram matrix of the generated image G
            - use only a single hidden layer a[l]
            - corresponding style cost for the layer

            - G(S)gram gram matrix of the style image
            - G(G)gram gram matrix of the generated image
            - the cost is computed using the hidden layer activations for a particular hidden layer in the network a[l]
            - Exercise 3 - compute_layer_style_cost
                - retrieve dimensions from the hidden layer activation a_G
                    - X.get_shape().as_list()
                - unroll the hidden layer activations a_s and a_G into 2D matrices
                    - use tf.transpose and tf.reshape
                - compute the style matrix of the images S and G
                - compute the style cost
                    - use tf.reduce_sum, tf.square, and tf.subtract
                - since the activation dimensions are (m, nh, nw, nc) whereas the desired unrolled matrix shape is (nc, nh * nw), the order of the filter dimension nc is changed
                - so tf.transpose can be used to change the order of the filter dimension

                ```python
                # UNQ_C3
                # GRADED FUNCTION: compute_layer_style_cost

                def compute_layer_style_cost(a_S, a_G):
                    """
                    Arguments:
                    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
                    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

                    Returns: 
                    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
                    """
                    ### START CODE HERE

                    # Retrieve dimensions from a_G (≈1 line)
                    _, n_H, n_W, n_C = a_G.shape

                    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W) (≈2 lines)
                    a_S = tf.reshape(tf.transpose(a_S, perm=[0,3,1,2]), shape=[n_C,-1])
                    a_G = tf.reshape(tf.transpose(a_G, perm=[0,3,1,2]), shape=[n_C,-1])

                    # Computing gram_matrices for both images S and G (≈2 lines)
                    GS = gram_matrix(a_S)
                    GG = gram_matrix(a_G)

                    # Computing the loss (≈1 line)
                    J_style_layer = (1 / (4 * (n_C ** 2) * ((n_H * n_W) ** 2))) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

                    ### END CODE HERE

                    return J_style_layer
                ```

    - Style Weights
        - captured the style from only one layer
        - better results if style costs are merged from several different layers
        - each layer will be given weights (lambda[l]) that reflect how much each layer will contribute to the style
        - be default, give each layer equal weight, and the weights add up to 1
        - list the layer names

            ```python
            for layer in vgg.layers:
                print(layer.name)

            '''
            input_1
            block1_conv1
            block1_conv2
            ...
            block5_conv3
            block5_conv4
            block5_poo
            '''
            ```

        - look at the output of a layer block_conv4
        - later define this as the content layer, which will represent the image

            ```python
            vgg.get_layer('block5_conv4').output

            # <KerasTensor: shape=(None, 25, 25, 512) dtype=float32 (created by layer 'block5_conv4')>
            ```

        - choose layers to represent the style of the image and assign style costs

            ```python
            STYLE_LAYERS = [
                ('block1_conv1', 0.2),
                ('block2_conv1', 0.2),
                ('block3_conv1', 0.2),
                ('block4_conv1', 0.2),
                ('block5_conv1', 0.2)]
            ```

        - combine the style costs for different layers

        - where the values for lambda[l] are given in STYLE_LAYERS
        - Exercise 4 - compute_style_cost
            - the compute_style_cost() function calls the compute_layer_style_cost() several times, and weights their results using the values in STYLE_LAYERS
            - for each layer
                - select the activation of the current layer
                - get the style of the style image S from the current layer
                - get the style of the generated image G from the current layer
                - compute the style cost for the current layer
                - add the weighted style cost to the overall style cost (J_style)
            - return the overall style cost

            ```python
            def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
                """
                Computes the overall style cost from several chosen layers

                Arguments:
                style_image_output -- our tensorflow model
                generated_image_output --
                STYLE_LAYERS -- A python list containing:
                                    - the names of the layers we would like to extract style from
                                    - a coefficient for each of them

                Returns: 
                J_style -- tensor representing a scalar value, style cost defined above by equation (2)
                """

                # initialize the overall style cost
                J_style = 0

                # Set a_S to be the hidden layer activation from the layer we have selected.
                # The last element of the array contains the content layer image, which must not be used.
                a_S = style_image_output[:-1]

                # Set a_G to be the output of the choosen hidden layers.
                # The last element of the list contains the content layer image which must not be used.
                a_G = generated_image_output[:-1]
                for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
                    # Compute style_cost for the current layer
                    J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

                    # Add weight * J_style_layer of this layer to overall style cost
                    J_style += weight[1] * J_style_layer

                return J_style
            ```

        - how to choose the coefficients for each layer?
        - the deeper layers capture higher-level concepts, and the features in the deeper layers are less localized in the image relative to each other
        - so for the generated image to softly follow style image, choose larger weights for deeper layers and smaller weights for the first layers
        - for the generated image to more strongly follow the style image, choose smaller weights for deeper layers and larger weights for the first layers
- Defining the Total Cost to Optimize
    - create a cost function that minimized both the style and the content cost

    - Exercise 5 - total_cost
        - implement the total cost function which includes both the content cost and the style cost

        ```python
        # UNQ_C4
        # GRADED FUNCTION: total_cost
        @tf.function()
        def total_cost(J_content, J_style, alpha = 10, beta = 40):
            """
            Computes the total cost function

            Arguments:
            J_content -- content cost coded above
            J_style -- style cost coded above
            alpha -- hyperparameter weighting the importance of the content cost
            beta -- hyperparameter weighting the importance of the style cost

            Returns:
            J -- total cost as defined by the formula above.
            """
            ### START CODE HERE

            #(≈1 line)
            J = alpha * J_content + beta * J_style

            ### START CODE HERE

            return J
        ```

Solving the Optimization Problem

- steps
    - load the content image
    - load the style image
    - randomly initialize the image to be generated
    - load the vgg19 model
    - compute the content cost
    - compute the style cost
    - compute the total cost
    - define the optimizer and learning rate
- Load the Content Image
    - load, reshape, and normalize the content image C

        ```python
        content_image = np.array(Image.open("images/louvre_small.jpg").resize((img_size, img_size)))
        content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

        print(content_image.shape)
        imshow(content_image[0])
        plt.show()

        # (1, 400, 400, 3)
        ```

- Load the Style Image
    - load, reshape and normalize the style image

        ```python
        style_image =  np.array(Image.open("images/monet.jpg").resize((img_size, img_size)))
        style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

        print(style_image.shape)
        imshow(style_image[0])
        plt.show()

        # (1, 400, 400, 3)
        ```

- Randomly Initialize the Image to be Generated
    - initialize the generated image as a noisy image created from the content_image
    - the generated image is slightly correlated with the content image
    - by initializing the pixels of the generated image to be mostly noise but slightly correlated with the content image, this will help the content of the generated image more rapidly match the content of the content image

        ```python
        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
        generated_image = tf.add(generated_image, noise)
        generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

        print(generated_image.shape)
        imshow(generated_image.numpy()[0])
        plt.show()

        # (1, 400, 400, 3)
        ```

- Load Pre-trained VGG19 Model
    - define a function which loads the vgg19 model and returns a list of the outputs for the middle layers

        ```python
        def get_layer_outputs(vgg, layer_names):
            """ Creates a vgg model that returns a list of intermediate output values."""
            outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

            model = tf.keras.Model([vgg.input], outputs)
            return model
        ```

    - define the content layer and build the model

        ```python
        content_layer = [('block5_conv4', 1)]

        vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
        ```

    - save the outputs for the content and style layers in separate variables

        ```python
        content_target = vgg_model_outputs(content_image)  # Content encoder
        style_targets = vgg_model_outputs(style_image)     # Style encoder
        ```

- Compute Total Cost
    - Compute the Content image Encoding (a_C)
        - compute the content cost
        - encode the content image using the appropriate hidden layer activations
        - set this encoding to the variable a_C
        - later, do the same for the generated image, by setting the variable a_G to the appropriate hidden layer activations
        - use layer block5_conv4 to compute the encoding
        - Set a_C to be the tensor giving he hidden layer activation for layer block5_cov4 using the content image

            ```python
            # Assign the content image to be the input of the VGG model.  
            # Set a_C to be the hidden layer activation from the layer we have selected
            preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
            a_C = vgg_model_outputs(preprocessed_content)
            ```

    - Compute the Style Image Encoding
        - set a_S to be the tensor giving the hidden layer activation for STYLE_LAYERS using the style image

            ```python
            # Assign the input of the model to be the "style" image 
            preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
            a_S = vgg_model_outputs(preprocessed_style)
            ```

        - some utils needed to display the images generated by the style transfer model

            ```python
            def clip_0_1(image):
                """
                Truncate all the pixels in the tensor to be between 0 and 1

                Arguments:
                image -- Tensor
                J_style -- style cost coded above

                Returns:
                Tensor
                """
                return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

            def tensor_to_image(tensor):
                """
                Converts the given tensor into a PIL image

                Arguments:
                tensor -- Tensor

                Returns:
                Image: A PIL image
                """
                tensor = tensor * 255
                tensor = np.array(tensor, dtype=np.uint8)
                if np.ndim(tensor) > 3:
                    assert tensor.shape[0] == 1
                    tensor = tensor[0]
                return Image.fromarray(tensor)
            ```

        - Exercise 6 - train_step
            - use the adam optimizer to minimize the total cost J
            - use a learning rate of 0.01
            - use tf.GradientTape to update the image
            - within the tf.GradientTape()
                - compute the encoding of the generated image using vgg_model_outputs
                - assign the result to a_G
                - compute the total cost J, using the global variables a_C, a_S, and local a_G
                - use alpha = 10, beta = 40

            ```python
            # UNQ_C5
            # GRADED FUNCTION: train_step

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            @tf.function()
            def train_step(generated_image):
                with tf.GradientTape() as tape:
                    # In this function you must use the precomputed encoded images a_S and a_C

                    ### START CODE HERE

                    # Compute a_G as the vgg_model_outputs for the current generated image
                    #(1 line)
                    a_G = vgg_model_outputs(generated_image)

                    # Compute the style cost
                    #(1 line)
                    J_style = compute_style_cost(a_S, a_G)

                    #(2 lines)
                    # Compute the content cost
                    J_content = compute_content_cost(a_C, a_G)
                    # Compute the total cost
                    J = total_cost(J_content, J_style)

                    ### END CODE HERE

                grad = tape.gradient(J, generated_image)

                optimizer.apply_gradients([(grad, generated_image)])
                generated_image.assign(clip_0_1(generated_image))
                # For grading purposes
                return J
            ```

- Train the Model
    - generate an artistic image
    - increasing learning rate speeds up the style transfer, but at the cost of quality

        ```python
        # Show the generated image at some epochs
        # Uncomment to reset the style transfer process. You will need to compile the train_step function again 
        epochs = 2501
        for i in range(epochs):
            train_step(generated_image)
            if i % 250 == 0:
                print(f"Epoch {i} ")
            if i % 250 == 0:
                image = tensor_to_image(generated_image)
                imshow(image)
                image.save(f"output/image_{i}.jpg")
                plt.show() 
        ```

        ```python
        # Show the 3 images in a row
        fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(1, 3, 1)
        imshow(content_image[0])
        ax.title.set_text('Content image')
        ax = fig.add_subplot(1, 3, 2)
        imshow(style_image[0])
        ax.title.set_text('Style image')
        ax = fig.add_subplot(1, 3, 3)
        imshow(generated_image[0])
        ax.title.set_text('Generated image')
        plt.show()
        ```

    - these results are from training the model with a learning rate of 0.01 and ran it for epochs = 2501
    - to get better looking results, running the optimization alg longer, 20000 epochs and lr = 0.001