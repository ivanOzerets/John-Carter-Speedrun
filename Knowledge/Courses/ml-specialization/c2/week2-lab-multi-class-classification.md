Multi-class Classification

- NN are often used to classify data
    - take in photos and classify subject in the photos as {dog, cat, horse, other}
    - take in a sentence and classify the ‘parts of speech’ of its elements: {nouns, verb, adjective, etc.}
- A network of this type will have multiple units in its final layer
- each output is associated with a categoty
- when an input example is applied to the network, the output with the highest value is the category predicted
- if the output is applied to a softmax function, the output of the softmax will provide probabilities of the input being in each category

Prepare and visualize the data

- will be using Scikit-Learn make_blobs function to make a training data set with four categories

    ```python
    # make 4-class dataset for classification
    classes = 4
    m = 100
    centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
    std = 1.0
    X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)
    ```

- each dot represents a training example
- the axis (x0, x1) are the inputs and the color represent the class the example is associated with
- once trained, the model will be presented with a new example (x0, x1), and will predict the class
- while generated, the data set is representative of many real-world classification problems
- there are several input features (x0,..,xn) and several output categories
- the model is trained to use the input features to predict the correct output category

    ```python
    # show classes in data set
    print(f"unique classes {np.unique(y_train)}")
    # show how classes are represented
    print(f"class representation {y_train[:10]}")
    # show shapes of our dataset
    print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")

    '''
    unique classes [0 1 2 3]
    class representation [3 3 3 0 3 3 3 3 2 0]
    shape of X_train: (100, 2), shape of y_train: (100,)
    '''
    ```

Model

- will use a 2-layer network
- unlike the binary classification networks, this network has four outputs, one for each class
- given an input example, the output with the highest value is the predicted class of the input
- example of how to construct this network in tensorflow
- the output layer uses a linear rather than a softmax activation
- while is is possible to include the softmax in the output layer, it is more numerically stable is linear outputs are passed to the loss function during training
- if the model is used to predict probabilities, the softmax can be applied at that point

    ```python
    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            Dense(2, activation = 'relu',   name = "L1"),
            Dense(4, activation = 'linear', name = "L2")
        ]
    )
    ```

- the statements below compile and train the network
- setting from_logits=True as an argument to the loss function specifies that the output activation was linear rather than a softmax

    ```python
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    model.fit(
        X_train,y_train,
        epochs=200
    )

    '''
    Epoch 1/200
    4/4 [==============================] - 0s 1ms/step - loss: 1.8158
    Epoch 2/200
    4/4 [==============================] - 0s 1ms/step - loss: 1.6976
    ...
    Epoch 199/200
    4/4 [==============================] - 0s 980us/step - loss: 0.0303
    Epoch 200/200
    4/4 [==============================] - 0s 1ms/step - loss: 0.0300
    <keras.callbacks.History at 0x7f9fbdaa9e90>
    '''
    ```

- with the model trained, we can see how the model has classified the training data

- the decision boundaries show how the model has partitioned the input space
- the model has had no trouble classifying the training data
- pull the trained weights from the model and use that to plot the function of each of the network units

    ```python
    # gather the trained parameters from the first layer
    l1 = model.get_layer("L1")
    W1,b1 = l1.get_weights()
    ```

    ```python
    # gather the trained parameters from the output layer
    l2 = model.get_layer("L2")
    W2, b2 = l2.get_weights()
    # create the 'new features', the training examples after L1 transformation
    Xl2 = np.zeros_like(X_train)
    Xl2 = np.maximum(0, np.dot(X_train,W1) + b1)

    plt_output_layer_linear(Xl2, y_train.reshape(-1,), W2, b2, classes,
                            x0_rng = (-0.25,np.amax(Xl2[:,0])), x1_rng = (-0.25,np.amax(Xl2[:,1])))
    ```

Explination

- Layer 1
    - the plots show the function of units 0 and 1 in the first layer of the network
    - the input are (x0,x1) on the axis
    - the output of the unit is represented by the color of the background
    - indicated by the color bar on the right of each graph
    - since these units are using a ReLU, the outputs do not necessarily fall between - and 1 and in this case are greated than 20 at their peaks
    - the contour lines in the graph shown the transition point between the output, a_j[1] being zero and non-zero
    - the contour line in the graph is the inflection point in the ReLU
    - unit 0 has separated classes 0 and 1 from classes 2 and 3
    - points to the left of the line (classes 0 and 1) will output zero, while points to the right will output a value greater than zero
    - unit 1 has separated classes 0 and 2 from classes 1 ad 3
    - points below will output a value grater than zero
- Layer 2
    - the dots in these graphs are the training examples translated by the first layer
    - one way to think of this is the first layer has created a new set of features for evaluation by the 2nd layer
    - the axes in these plots are the outputs of the previous layer a_0^[1] and a_1^[1]
    - classes 0 and 1 (green and blue) have a_0^[i] = 0 while classes 1 and 2 (blue and orange) have a_1^[1] = 0
    - once again, the intensity of the background color indicates the highest values
    - unit 0 will produce its maximum value for values near (0,0), where class 0 (blue) has been mapped
    - unit 1 produces its highest values in the upper left corner selecting class 1 (green)
    - unit 2 targets the lower right corner where class 3 (orange) resides
    - unit 3 produces its highest values in the upper right selecting the final class (purple)
    - one other aspect that is not obvious from the graphs is that the values have been coordinated between units
    - it is not sufficient for a unit to produce a maximum value for the class it is selected for, it must also be the highest value of all the units for points in that class
    - that is done by implied softmax function that is part of the loss function (SparseCategoricalCrossEntropy)
    - unlike other activation functions, softmax works across all the outputs
    - you can successfully use NN without knowing the details of what each unit is up to