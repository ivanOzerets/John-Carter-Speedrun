Image Segmentation with U-Net

- building a U-Net, a type of CNN designed for quick, precise image segmentation, and using it to predict a label for every single pixel in an image
- this type of image classification is called semantic image segmentation
- it’s similar to object detection in that both ask the question: what objects are in this image and where in the image are those objects located
- but where object detection labels objects with bounding boxes that may include pixels that aren’t part of the object, semantic image segmentation allows to predict a precise mask for each object in the image by labeling each pixel int eh image with its corresponding class
- the word semantic here refers to what’s being shown, so for example the car class is indicated below by the dark blue mask, and person is indicated with a red mask

- region-specific labeling is a pretty crucial consideration for self-driving cars, which require a pixel-perfect understanding of their environment so they can change lanes and avoid other cars, or any number of traffic obstacles that can put peoples’ lives in danger

Packages

```python
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

from test_utils import summary, comparator
```

Load and Split the Data

```python
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio

import matplotlib.pyplot as plt
%matplotlib inline

path = ''
image_path = os.path.join(path, './data/CameraRGB/')
mask_path = os.path.join(path, './data/CameraMask/')
image_list_orig = os.listdir(image_path)
image_list = [image_path+i for i in image_list_orig]
mask_list = [mask_path+i for i in image_list_orig]
```

- check out some of the unmasked and masked images from the dataset

    ```python
    N = 2
    img = imageio.imread(image_list[N])
    mask = imageio.imread(mask_list[N])
    #mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

    fig, arr = plt.subplots(1, 2, figsize=(14, 10))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[1].imshow(mask[:, :, 0])
    arr[1].set_title('Segmentation')
    ```

- Split Your Dataset into Unmasked and Masked Images

    ```python
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

    for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
        print(path)
    ```

    ```python
    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    for image, mask in dataset.take(1):
        print(image)
        print(mask)
    ```

- Preprocess Your Data
    - normally, normalize the image values by dividing them by 255
    - this sets them between 0 and 1
    - however, using tf.image.convert_image_dtype with tf.float32 sets them between 0 and 1
    - so there’s no need to further divide them by 255

        ```python
        def process_path(image_path, mask_path):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)

            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=3)
            mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
            return img, mask

        def preprocess(image, mask):
            input_image = tf.image.resize(image, (96, 128), method='nearest')
            input_mask = tf.image.resize(mask, (96, 128), method='nearest')

            return input_image, input_mask

        image_ds = dataset.map(process_path)
        processed_image_ds = image_ds.map(preprocess)
        ```

U-Net

- named for its U-shape
- originally created in 2015 for tumor detection, but in years since has become a very popular choice for other semantic segmentation tasks
- U-Net builds on a previous architecture called the Fully Convolutional Network, or FCN, which replaces the dense layers found in a typical CNN with a transposed convolution layer that up samples the feature map back to the size of the original input image, while preserving the spatial information
- this is necessary because the dense layers destroy spatial information, which is an essential part of image segmentation tasks
- an added bonus of using transpose convs is that the input size no longer needs to be fixes, as it does when dense layers are used
- unfortunately, the final feature layer of the FCN suffers from information loss due to downsampling too much
- it then becomes difficult to upsample after so much information has been lost, causing an output that looks rough
- U-Net improves on the RFCN, using a somewhat similar design, but differing in some important ways
- instead of one transposed conv at the end of the network, it uses a matching number of convs for downsampling th input image to a feature map, and transposed convs for upsampling those amps back to the orignal input image size
- it also adds skip connections, to retain information that would otherwise become lost during encoding
- skip connections send information to every upsampling layer in the decoder form the corresponding downsampling layer in the encoder, capturing finer information while also keeping computation low
- these help prevent information loss, as ell as model overfilling
- Model Details

    - Contracting path
        - images are first few through several convolutional layers which reduce height and width, while growing the numebr of channels
        - the contracting path follows a regular CNN architecture, with conv layers, their activations ,and pooling layers to downsample the image and extract its features
        - in detail, it consists of the repeated application of two 3x3 same padding convolutions, each followed by a relu and a 2x2 max pooling opearation with stride 2 for downsampling
        - at each downsampling step the number of feature channels is doubles
    - Crop function
        - this step crops the image from the contracting path and concatenates it to the current image on the expanding path o create a skip connection
        - why crop?
    - Expanding path (Decoder containing upsampling steps
        - the expanding path performs the opposite operation of the contracting path, growing the image back to its original size, while shrinking the channels gradually
        - in detail, each step in the expanding path upsamples the feature map, followed by a 2x2 conv (transposed conv)
        - this transposed conv halves the number of feature channels, while growing the height and width of the image
        - next is the concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convs, each followed by a relu
        - need to perform cropping to handle the loss of border pixels in every convolution
    - Final Feature Mapping Block
        - in the final layer, a 1x1 conv is used to map each 64-component feature vector to the desired number of classes
        - the channel dimensions from the previous layer correspond to the number of filters used, so when using 1x1 convs, you can transform that dimension by choosing an appropriate number of 1x1 filters
        - when this idea is applied to the last layer, reduce the channel dimensions to have one layer per class
        - the U-Net network has 23 conv layers in total
- Encoder (Downsampling Block)

    - the encoder is a stack of various conv_blocks
    - each conv_block() is composed of 2 Conv2D layers with relu activations
    - apply Dropout, and MaxPooling2D to some conv_blocks, specifically to the last two blocks of the downsampling
    - the function will return two tensors
        - next_layer: that will go into the next block
        - skip_connection: that will go into the corresponding decoding block’
    - if max_pooling=True, the next_layer will be the output of the MaxPooling2D layer, but the skip_connection will be the output of the previously applied layer (conv2d or dropout, depending on the case)
    - else, both results will be identical
    - Exercise 1 - conv_block
        - implement conv_block(..)
        - aded 2 conv2d layers with n_filters filters with kernel_size set to 3, kernel_initializer set to ‘he_normal’, padding set to ‘same’ and ‘relu’ activation
        - if dropout_prob > 0, then add a dropout layer with parameters dropout_prob
        - if max_pooling is set to True, then add a MaxPooling2D layer with 2x2 pool size

        ```python
        # UNQ_C1
        # GRADED FUNCTION: conv_block
        def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
            """
            Convolutional downsampling block

            Arguments:
                inputs -- Input tensor
                n_filters -- Number of filters for the convolutional layers
                dropout_prob -- Dropout probability
                max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
            Returns: 
                next_layer, skip_connection --  Next layer and skip connection outputs
            """

            ### START CODE HERE
            conv = Conv2D(n_filters, # Number of filters
                          3,   # Kernel size   
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(inputs)
            conv = Conv2D(n_filters, # Number of filters
                          3,   # Kernel size
                          activation='relu',
                          padding='same',
                          # set 'kernel_initializer' same as above
                          kernel_initializer='he_normal')(conv)
            ### END CODE HERE

            # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
            if dropout_prob > 0:
                 ### START CODE HERE
                conv = Dropout(dropout_prob)(conv)
                 ### END CODE HERE

            # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
            if max_pooling:
                ### START CODE HERE
                next_layer = MaxPooling2D(pool_size=(2,2))(conv)
                ### END CODE HERE

            else:
                next_layer = conv

            skip_connection = conv

            return next_layer, skip_connection

        '''
        Block 1:
        ['InputLayer', [(None, 96, 128, 3)], 0]
        ['Conv2D', (None, 96, 128, 32), 896, 'same', 'relu', 'HeNormal']
        ['Conv2D', (None, 96, 128, 32), 9248, 'same', 'relu', 'HeNormal']
        ['MaxPooling2D', (None, 48, 64, 32), 0, (2, 2)]
        All tests passed!

        Block 2:
        ['InputLayer', [(None, 96, 128, 3)], 0]
        ['Conv2D', (None, 96, 128, 1024), 28672, 'same', 'relu', 'HeNormal']
        ['Conv2D', (None, 96, 128, 1024), 9438208, 'same', 'relu', 'HeNormal']
        ['Dropout', (None, 96, 128, 1024), 0, 0.1]
        ['MaxPooling2D', (None, 48, 64, 1024), 0, (2, 2)]
        All tests passed!
        '''
        ```

- Decoder (Upsampling Block)
    - the decoder, or upsampling block, upsamples the features back to the original image size
    - at each upsampling level, take the output of the corresponding encoder block and concatenate it before feeding to the next decoder block
    - there are two new component in the decoder: up and merge
    - these are the transpose conv and the skip connections
    - in addition, there are two more conv layers set to the same params as in the encoder
    - encounter the conv2dtranspose layer, which performs the inverse of the conv2d layer
    - Exercise 2 - upsampling_block
        - implement upsampling_block(…)
        - takes the args expansive_input (which is the input tensor from the previous layer) and contractive_input (the input tensor from the previous skip layer)
        - the number of filters here is the same as in the downsampling block completed previously
        - the conv2dtranspose layer will take n_filters with shape (3,3) and a stride of (2,2) with padding set to same
        - it’s applied to expansive_input, or the input tensor from the previous layer
        - this block is also where you concatenate the outputs from the encoder blocks, creating skip connections
        - concatenate the conv2dtranspose layer output to the contractive input, with an axis of 3
        - in general, concatenate the tensors in any order of preference, but for this exercise use [up, contractive_input]
        - for the final component, se the params for two conv2d layers to the same values used for the two conv2d layers int he encoder (relu activation, he normal initializer, same padding)

        ```python
        # UNQ_C2
        # GRADED FUNCTION: upsampling_block
        def upsampling_block(expansive_input, contractive_input, n_filters=32):
            """
            Convolutional upsampling block

            Arguments:
                expansive_input -- Input tensor from previous layer
                contractive_input -- Input tensor from previous skip layer
                n_filters -- Number of filters for the convolutional layers
            Returns: 
                conv -- Tensor output
            """

            ### START CODE HERE
            up = Conv2DTranspose(
                         n_filters,    # number of filters
                         3,    # Kernel size
                         strides=2,
                         padding='same')(expansive_input)

            # Merge the previous output and the contractive_input
            merge = concatenate([up, contractive_input], axis=3)
            conv = Conv2D(n_filters,   # Number of filters
                         3,     # Kernel size
                         activation='relu',
                         padding='same',
                         kernel_initializer='he_normal')(merge)
            conv = Conv2D(n_filters,  # Number of filters
                         3,   # Kernel size
                         activation='relu',
                         padding='same',
                          # set 'kernel_initializer' same as above
                         kernel_initializer='he_normal')(conv)
            ### END CODE HERE

            return conv

        '''
        Block 1:
        ['InputLayer', [(None, 12, 16, 256)], 0]
        ['Conv2DTranspose', (None, 24, 32, 32), 73760]
        ['InputLayer', [(None, 24, 32, 128)], 0]
        ['Concatenate', (None, 24, 32, 160), 0]
        ['Conv2D', (None, 24, 32, 32), 46112, 'same', 'relu', 'HeNormal']
        ['Conv2D', (None, 24, 32, 32), 9248, 'same', 'relu', 'HeNormal']
        '''
        ```

- Build the Model
    - chain the encode, bottleneck, and decoder
    - need to specify the number of output channels, which will be set to 23
    - that’s because there are 23 possible labels for each pixel in this self-driving car dataset
    - Exercise 3 - unet_model
        - for the function unet_model, specify the input shape, number of filters, and number of classes
        - for the first half of the model
            - begin with a conv block that takes the inputs of the model and the number of filters
            - then, chain the first output element of each block to the input of the next conv block
            - next, double the number of filters at each step
            - beginning with conv_block, set dropout_prob to 0.3 again, and turn off max pooling
        - for the second half
            - use cblock5 as expansive_input and cblock4 as contractive_input, with n_filters * 8 - this is the bottleneck layer
            - chain the output of the previous block as expansive_input and the corresponding contractive block output
            - note to use the second element of the contractive block before the max pooling layer
            - at each step, use half the number of filters oft he previous block
            - conv9 is a conv2d layer with relu activation, he normal initializer, same padding
            - finally, conv10 is a conv2d that takes the number of classes as the filter, a kernel size of 1, and ‘same’ padding
            - the output of the conv10 is the output of the model

        ```python
        # UNQ_C3
        # GRADED FUNCTION: unet_model
        def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
            """
            Unet model

            Arguments:
                input_size -- Input shape 
                n_filters -- Number of filters for the convolutional layers
                n_classes -- Number of output classes
            Returns: 
                model -- tf.keras.Model
            """
            inputs = Input(input_size)
            # Contracting Path (encoding)
            # Add a conv_block with the inputs of the unet_ model and n_filters
            ### START CODE HERE
            cblock1 = conv_block(inputs, n_filters)
            # Chain the first element of the output of each block to be the input of the next conv_block. 
            # Double the number of filters at each new step
            cblock2 = conv_block(cblock1[0], n_filters * 2)
            cblock3 = conv_block(cblock2[0], n_filters * 4)
            cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3) # Include a dropout_prob of 0.3 for this layer
            # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
            cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False) 
            ### END CODE HERE

            # Expanding Path (decoding)
            # Add the first upsampling_block.
            # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
            ### START CODE HERE
            ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
            # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
            # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
            # At each step, use half the number of filters of the previous block 
            ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
            ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
            ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)
            ### END CODE HERE

            conv9 = Conv2D(n_filters,
                         3,
                         activation='relu',
                         padding='same',
                         # set 'kernel_initializer' same as above exercises
                         kernel_initializer='he_normal')(ublock9)

            # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
            ### START CODE HERE
            conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
            ### END CODE HERE

            model = tf.keras.Model(inputs=inputs, outputs=conv10)

            return model
        ```

- Set Model Dimensions

    ```python
    img_height = 96
    img_width = 128
    num_channels = 3

    unet = unet_model((img_height, img_width, num_channels))
    ```

    ```python
    unet.summary()

    '''
    Model: "model_4"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_7 (InputLayer)           [(None, 96, 128, 3)  0           []                               
                                    ]                                                                 

     conv2d_32 (Conv2D)             (None, 96, 128, 32)  896         ['input_7[0][0]']                

     conv2d_33 (Conv2D)             (None, 96, 128, 32)  9248        ['conv2d_32[0][0]']              

     max_pooling2d_9 (MaxPooling2D)  (None, 48, 64, 32)  0           ['conv2d_33[0][0]']              

     conv2d_34 (Conv2D)             (None, 48, 64, 64)   18496       ['max_pooling2d_9[0][0]']     
     ...
      conv2d_48 (Conv2D)             (None, 96, 128, 32)  18464       ['concatenate_8[0][0]']          

     conv2d_49 (Conv2D)             (None, 96, 128, 32)  9248        ['conv2d_48[0][0]']              

     conv2d_50 (Conv2D)             (None, 96, 128, 32)  9248        ['conv2d_49[0][0]']              

     conv2d_51 (Conv2D)             (None, 96, 128, 23)  759         ['conv2d_50[0][0]']              

    ==================================================================================================
    Total params: 8,640,471
    Trainable params: 8,640,471
    Non-trainable params: 0
    __________________________________________________________________________________________________
    '''
    ```

- Loss Function
    - in semantic segmentation, you need as many masks as you have object classes
    - in the dataset, each pixel in every mask has been assigned a single integer probability that it belongs to a certain class, from 0 to num_classes_1
    - the correct class is the layer with the higher probability
    - this is different from categorical crossentropy, where the labels should be one0hot encoded
    - here, use sparse categorical crossentropy as the loss function, to perform pixel-wise multiclass prediction
    - sparse categorical crossentropy is more efficient than other loss functions when dealing with lots of classes

        ```python
        unet.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        ```

- Dataset Handling
    - define a function that displays both an input image, and its ground truth, the true mask
    - the true mask is what the trained model output is aiming to get as close to as possible

        ```python
        def display(display_list):
            plt.figure(figsize=(15, 15))

            title = ['Input Image', 'True Mask', 'Predicted Mask']

            for i in range(len(display_list)):
                plt.subplot(1, len(display_list), i+1)
                plt.title(title[i])
                plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
                plt.axis('off')
            plt.show()
        ```

        ```python
        for image, mask in image_ds.take(1):
            sample_image, sample_mask = image, mask
            print(mask.shape)
        display([sample_image, sample_mask])

        # (80, 640, 1)
        ```

        ```python
        for image, mask in processed_image_ds.take(1):
            sample_image, sample_mask = image, mask
            print(mask.shape)
        display([sample_image, sample_mask])

        # (96, 128, 1)
        ```

Train the Model

```python
EPOCHS = 5
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)
model_history = unet.fit(train_dataset, epochs=EPOCHS)

'''

'''
```

- Create Predicted Masks
    - define a function that uses tf.argmax in the axis of the number of classes to return the index with the largest value and merge the prediction into a single image

        ```python
        def create_mask(pred_mask):
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]
            return pred_mask[0]
        ```

- Plot Model Accuracy

    ```python
    plt.plot(model_history.history["accuracy"])
    ```

- Show Predications
    - check the predicted masks against the true mask and the original input image

        ```python
        def show_predictions(dataset=None, num=1):
            """
            Displays the first image of each of the num batches
            """
            if dataset:
                for image, mask in dataset.take(num):
                    pred_mask = unet.predict(image)
                    display([image[0], mask[0], create_mask(pred_mask)])
            else:
                display([sample_image, sample_mask,
                     create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
        ```

        ```python
        show_predictions(train_dataset, 6)
        ```

    - with just 40 epochs

Training code to get consistent results

```python
EPOCHS = 15
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)

unet = unet_model((img_height, img_width, num_channels))
unet.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model_history = unet.fit(train_dataset, epochs=EPOCHS)
```