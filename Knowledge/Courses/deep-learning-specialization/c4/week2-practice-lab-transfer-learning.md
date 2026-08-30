Packages

```python
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
```

- Create the dataset and Split it into Training and Validation Sets
    - when training and evaluating deep learning models in keras, generating a dataset from image files stored on disk is simple and fast
    - call image_data_set_from_directory() to read from the directory and create both training and validation datasets
    - when specifying a validation split, also specify the subset for each portion
    - just set the training set to subset=’training’ and the validation to subset=’validation’
    - set the seeds to match each other, so the trianing and validation sets don’t overlap

    ```python
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)
    directory = "dataset/"
    train_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 validation_split=0.2,
                                                 subset='training',
                                                 seed=42)
    validation_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 validation_split=0.2,
                                                 subset='validation',
                                                 seed=42)

    '''
    Found 327 files belonging to 2 classes.
    Using 262 files for training.
    Found 327 files belonging to 2 classes.
    Using 65 files for validation.
    '''
    ```

- look at the images from the training set
- the original dataset has some mislabeled images in it as well

    ```python
    class_names = train_dataset.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    ```

Preprocess and Augment Training Data

- using prefetch() prevents a memory bottleneck that can occur when reading from the disk
- it sets aside some data and keeps it ready for when it’s needed, by creating a source dataset from the input data, applying a transformation to preprocess it, then iterating over the dataset one element at a time
- because the iteration is streaming, the data doesn’t need to fit into memory
- can set the number of elements to prefetch manually, or use tf.data.experimental.AUTOTUNE to choose the parameters automatically
- autotune prompts [tf.data](http://tf.data) to tune that value dynamically at runtime, by tracking the time spent in each operation and feeding those times into an optimization alg
- the optimization alg tries to find the bast allocation of its CPU budget across all tunable operations
- to increase diversity in the training set and help the model learn the data better, it’s standard practice to augment the images by transforming them, i.e. randomly flipping and rotating them
- Keras’ Sequential API offers a straightforward method for these kinds of data augmentations, with built-in, customizable preprocessing layers
- these layers are saved with the rest of the model and can be reused later

    ```python
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    ```

- Exercise 1 - data_augmenter
    - implement a function for data augmentation
    - use a Sequential keras model composed of 2 layers
        - RandomFlip(’horizontal’)
        - RandomRotation(0.2)

        ```python
        # UNQ_C1
        # GRADED FUNCTION: data_augmenter
        def data_augmenter():
            '''
            Create a Sequential model composed of 2 layers
            Returns:
                tf.keras.Sequential
            '''
            ### START CODE HERE
            data_augmentation = tf.keras.Sequential()
            data_augmentation.add(RandomFlip('horizontal'))
            data_augmentation.add(RandomRotation(0.2))
            ### END CODE HERE

            return data_augmentation
        ```

    - look at how an image from the training set has been augmented with simple transformations
    - from one cute animal, to 9 variations of that cute animal, in three lines of code

        ```python
        data_augmentation = data_augmenter()

        for image, _ in train_dataset.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')
        ```

    - next, normalize the input
    - trained on the normalization values [-1,1], it’s best practice to reuse the standard with tf.keras.applications.mobilenet_v2.preprocess_input

        ```python
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        ```

Using MobileNetV2 for Transfer Learning

- MobileNetV2 was trained on ImageNet and is optimized to run on mobile and other low-power applications
- its 155 layers deep and very efficient for object detection and image segmentation tasks as well as classification tasks
- the architecture has three characteristics
    - depthwise separable convolutions
    - thin input and output bottlenecks between layers
    - shortcut connections between bottleneck layers
- Inside a MobileNetV2 Convolutional Building Block
    - MobileNetV2 uses depthwise separable conv as efficient building blocks
    - traditional convolutions are often very resource-intensive, and depthwise separable conv are able to reduce the number of trainable params and operations and also speed up conv in two steps
        - the first step calculates an intermediate result by convolving on each of the channels independently, depthwise conv
        - the second step, another conv merges the outputs of the previous step into one
        - this gets a single result from a single feature at a time, and then is applied to all the filters in the output layer
        - this is the pointwise cov, or: shape of the depthwise conv x number of filters

    - Each block consists of an inverted residual structure with a bottleneck at each end
    - the bottlenecks encode the intermediate inputs and outputs in a low dimensional space, and prevent non-linearities from destroying important information
    - the shortcut connections, which are similar to the ones in traditional residual networks, serve the same purpose of speeding up training and improving predictions
    - these connections skip over the intermediate conv and connect the bottleneck layers
    - train the base model using all the layers from the pretrained model
    - similarly to reusing the pretrained normalization values MobileNetV2 was trained on, also load the pretrained weights from imageNet
    - by specifying weights=’imagenet’, the weights are downloaded for imageNet
    - load them locally from the workspace

        ```python
        IMG_SHAPE = IMG_SIZE + (3,)
        base_model_path="imagenet_base_model/with_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5"

        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=True,
                                                       weights=base_model_path)
        ```

    - print model summary for model layers, shape of the outputs, and total number of params, trainable and non-trainable

        ```python
        base_model.summary()

        '''
        Model: "mobilenetv2_1.00_160"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_1 (InputLayer)            [(None, 160, 160, 3) 0                                            
        __________________________________________________________________________________________________
        Conv1_pad (ZeroPadding2D)       (None, 161, 161, 3)  0           input_1[0][0]                    
        __________________________________________________________________________________________________
        Conv1 (Conv2D)                  (None, 80, 80, 32)   864         Conv1_pad[0][0]                  
        __________________________________________________________________________________________________
        bn_Conv1 (BatchNormalization)   (None, 80, 80, 32)   128         Conv1[0][0]                      
        __________________________________________________________________________________________________
        Conv1_relu (ReLU)               (None, 80, 80, 32)   0           bn_Conv1[0][0]                   
        __________________________________________________________________________________________________
        ...
        Conv_1 (Conv2D)                 (None, 5, 5, 1280)   409600      block_16_project_BN[0][0]        
        __________________________________________________________________________________________________
        Conv_1_bn (BatchNormalization)  (None, 5, 5, 1280)   5120        Conv_1[0][0]                     
        __________________________________________________________________________________________________
        out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]                  
        __________________________________________________________________________________________________
        global_average_pooling2d (Globa (None, 1280)         0           out_relu[0][0]                   
        __________________________________________________________________________________________________
        predictions (Dense)             (None, 1000)         1281000     global_average_pooling2d[0][0]   
        ==================================================================================================
        Total params: 3,538,984
        Trainable params: 3,504,872
        Non-trainable params: 34,112
        '''
        ```

    - the last two layers are so called top layers
    - they are responsible for the classification in the model

        ```python
        nb_layers = len(base_model.layers)
        print(base_model.layers[nb_layers - 2].name)
        print(base_model.layers[nb_layers - 1].name)

        '''
        global_average_pooling2d
        predictions
        '''
        ```

    - layers like conv2d and depthwiseconv2d follow the progression of expansion to depthwise conv to projection
    - in combination with batchnorm and relu, these make up the bottleneck layers\
    - choose the first batch from the tensorflow dataset to use the images, and run through the MobileNetV2 base model to test out the predictions on some images

        ```python
        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch)
        print(feature_batch.shape)

        # (32, 1000)
        ```

        ```python
        #Shows the different label probabilities in one tensor 
        label_batch

        '''
        <tf.Tensor: shape=(32,), dtype=int32, numpy=
        array([1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
               0, 0, 1, 0, 1, 1, 1, 1, 0, 0], dtype=int32)>
        '''
        ```

    - decode the predictions made by the model
    - the number 32 refers to the batch size and 1000 refers to the 1000 classes the model was pretrained on
    - the predictions returned by the base model follow the format
    - first the class number, then a human-readable label, and last the probability of the image belonging to that class
    - there are two of these returned for each image in the batch - these are the top two probabilities returned for that image

        ```python
        base_model.trainable = False
        image_var = tf.Variable(preprocess_input(image_batch))
        pred = base_model(image_var)

        # Function to decode predictions
        def decode_predictions(preds, top=2):
            results = []
            for pred in preds:
                top_indices = pred.argsort()[-top:][::-1]
                result = [tuple(class_index[str(i)]) + (pred[i],) for i in top_indices]
                results.append(result)
            return results

        with open("imagenet_base_model/imagenet_class_index.json", 'r') as f:
            class_index = json.load(f)

        decoded_predictions = decode_predictions(pred.numpy(), top=2)
        print(decoded_predictions)

        '''
        [[('n02489166', 'proboscis_monkey', 0.10329965), 
        ('n02102177', 'Welsh_springer_spaniel', 0.07883611)],
        ...
        [('n02437616', 'llama', 0.94477594), 
        ('n02423022', 'gazelle', 0.0054335156)]]
        '''
        ```

    - a lot of labels, some are very wrong
    - this is because MobilNet pretrained over ImageNet doesn’t have the correct labels for alpacas, so when using the full model, a bunch of incorrectly classified images are the result
- Layer Freezing with the Functional API
    - use a pretrained model to modify the classifier task to recognize alpacas
    - delete the top layer
        - set include_top in base_model as False
    - add a new classifier layer
        - train only one layer by freezing the rest of the network
        - a single neuron is enough to solve a binary classification problem
    - freeze the base model and train the newly-created classifier layer
        - set base model.trainable=False to avoid changing the weights and train only the new layer
        - set training in base_model to False to avoid keeping track of statistics in the batch norm layer
    - Exercise 2 - alpaca_model

        ```python
        # UNQ_C2
        # GRADED FUNCTION
        def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
            ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
            Arguments:
                image_shape -- Image width and height
                data_augmentation -- data augmentation function
            Returns:
            Returns:
                tf.keras.model
            '''

            input_shape = image_shape + (3,)

            ### START CODE HERE

            base_model_path="imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"

            base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                           include_top=False, # <== Important!!!!
                                                           weights=base_model_path)

            # freeze the base model by making it non trainable
            base_model.trainable = False 

            # create the input layer (Same as the imageNetv2 input size)
            inputs = tf.keras.Input(shape=input_shape) 

            # apply data augmentation to the inputs
            x = data_augmentation(inputs)

            # data preprocessing using the same weights the model was trained on
            x = preprocess_input(x) 

            # set training to False to avoid keeping track of statistics in the batch norm layer
            x = base_model(x, training=False) 

            # add the new Binary classification layers
            # use global avg pooling to summarize the info in each channel
            x = tfl.GlobalAveragePooling2D()(x) 
            # include dropout with probability of 0.2 to avoid overfitting
            x = tfl.Dropout(rate=0.2)(x)

            # use a prediction layer with one neuron (as a binary classifier only needs one)
            outputs = tfl.Dense(units=1)(x)

            ### END CODE HERE

            model = tf.keras.Model(inputs, outputs)

            return model
        ```

        ```python
        model2 = alpaca_model(IMG_SIZE, data_augmentation

        '''
        ['InputLayer', [(None, 160, 160, 3)], 0]
        ['Sequential', (None, 160, 160, 3), 0]
        ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0]
        ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0]
        ['Functional', (None, 5, 5, 1280), 2257984]
        ['GlobalAveragePooling2D', (None, 1280), 0]
        ['Dropout', (None, 1280), 0, 0.2]
        ['Dense', (None, 1), 1281, 'linear']
        '''
        ```

    - the base learning rate is 0.001
    - compile the model and run for 5 epochs

        ```python
        base_learning_rate = 0.001
        model2.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        ```

        ```python
        initial_epochs = 5
        history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

        '''
        Epoch 1/5
        9/9 [==============================] - 9s 1s/step - loss: 0.7903 - accuracy: 0.4924 - val_loss: 0.6703 - val_accuracy: 0.5385
        Epoch 2/5
        9/9 [==============================] - 7s 828ms/step - loss: 0.6148 - accuracy: 0.6412 - val_loss: 0.5459 - val_accuracy: 0.5692
        Epoch 3/5
        9/9 [==============================] - 8s 835ms/step - loss: 0.4903 - accuracy: 0.7405 - val_loss: 0.4443 - val_accuracy: 0.7077
        Epoch 4/5
        9/9 [==============================] - 8s 836ms/step - loss: 0.4721 - accuracy: 0.7366 - val_loss: 0.4257 - val_accuracy: 0.7231
        Epoch 5/5
        9/9 [==============================] - 8s 846ms/step - loss: 0.4040 - accuracy: 0.7863 - val_loss: 0.3703 - val_accuracy: 0.7846
        '''
        ```

    - plot the training and validation accuracy

        ```python
        acc = [0.] + history.history['accuracy']
        val_acc = [0.] + history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
        ```

        ```python
        class_names

        #['alpaca', 'not alpaca']
        ```

- Fine-tuning the Model
    - try fine-tuning the model by re-running the optimizer in the last layers to improve accuracy
    - when using a smaller learning rate, smaller steps are taken to adapt it a little more closely to the new data
    - in transfer learning, the way to achieve this is by unfreezing the layers at the end of the network, and then re-training the model on the final layers with a very low learning rate
    - adapting the learning rate to go over these layers in smaller steps can yield more fine details → a higher accuracy
    - when the network is in its earlier stages, it trains on low-level features, like edges
    - in the later layers, more complex, high-level features like wispy hair or pointy ears begin to emerge
    - for transfer learning, the low level features can be kept the same, as they have common features for most images
    - when adding new data, generally want the high-level features to adapt to it, which is rather like letting the network learn to detect features more related tot he data, such as soft fur or big teeth
    - to achieve this, just unfreeze the final layers and rerun the optimizer with a smaller learning rate, while keeping al the other layers frozen
    - where the final layers actually begin is a bit arbitrary
    - the important takeaway is that the later layers are the part of the network that contain the fine details that are more specific to the problem
    - first, unfreeze the base model by setting base_model.trainable=True, set a layer to fine-tune from, then re-freeze all the layers before it
    - run again and see the accuracy improve
    - Exercise 3

        ```python
        # UNQ_C3
        base_model = model2.layers[4]
        base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 120

        ### START CODE HERE

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Define a BinaryCrossentropy loss function. Use from_logits=True
        loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
        optimizer = tf.keras.optimizers.Adam(lr=0.1*base_learning_rate)
        # Use accuracy as evaluation metric
        metrics=['accuracy']

        ### END CODE HERE

        model2.compile(loss=loss_function,
                      optimizer = optimizer,
                      metrics=metrics)

        # Number of layers in the base model:  155
        ```

        ```python
        fine_tune_epochs = 5
        total_epochs =  initial_epochs + fine_tune_epochs

        history_fine = model2.fit(train_dataset,
                                 epochs=total_epochs,
                                 initial_epoch=history.epoch[-1],
                                 validation_data=validation_dataset)

        '''
        Epoch 5/10
        9/9 [==============================] - 10s 1s/step - loss: 0.6704 - accuracy: 0.7023 - val_loss: 0.3283 - val_accuracy: 0.9077
        Epoch 6/10
        9/9 [==============================] - 9s 1s/step - loss: 0.3923 - accuracy: 0.7824 - val_loss: 0.2005 - val_accuracy: 0.9231
        Epoch 7/10
        9/9 [==============================] - 9s 1s/step - loss: 0.3016 - accuracy: 0.8855 - val_loss: 0.2717 - val_accuracy: 0.8000
        Epoch 8/10
        9/9 [==============================] - 9s 1s/step - loss: 0.2315 - accuracy: 0.9046 - val_loss: 0.1049 - val_accuracy: 0.9385
        Epoch 9/10
        9/9 [==============================] - 9s 1s/step - loss: 0.1669 - accuracy: 0.9313 - val_loss: 0.0693 - val_accuracy: 0.9846
        Epoch 10/10
        9/9 [==============================] - 9s 978ms/step - loss: 0.1861 - accuracy: 0.9275 - val_loss: 0.0698 - val_accuracy: 0.9692
        '''
        ```

        ```python
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']
        ```

        ```python
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        plt.plot([initial_epochs-1,initial_epochs-1],
                  plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([initial_epochs-1,initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
        ```