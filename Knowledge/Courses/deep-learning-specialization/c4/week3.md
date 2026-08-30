Object Localization

- What are localization and detection
    - image classification
        - is this a car
    - classification with localization
        - is this a car and where is it
    - detection
        - multiple objects and detect them all

- Classification with localization
    - object categories or categories for softmax output
    - example
        - pedestrian
        - car
        - motorcycle
        - background
    - to localize the car in the image
    - change the NN to have more output units to output a bounded box
    - specifically four number that parameterize the bounding box
    - bx, by, bh, bw
    - (0,0) will be top left
    - bx and by denotes the center of the box

- Defining the target label y
    - classes
        - pedestrian
        - car
        - motorcycle
        - background
    - need to output bx, by, bh, bw, class label probs
    - Pc is the probability that an object is there
    - if so, output the bs
    - then which object, c1, c2, c3
    - if Pc is 0, then the other values are ‘don’t cares’

    - loss function is squared error on values if Pc is 1, only on Pc if Pc is 0

    - in practice log likelihood is used for c1, c2, c3
    - squared error for bs
    - and logistic loss for Pc

Landmark Detection

- a NN can output x and y coordinates of important points in an image (landmarks)
- Landmark detection
    - corners of an eye
    - l1x, l1y, l2x, l2y, …
    - how about mouth shape as well, or nose
    - define 64 landmarks on the face
    - convnet output into features → face?, l1x, l1y, …
    - used for filters for examples
    - labeled training set needed
    - also useful for people pose detection
    - landmarks must be consistent

Object Detection

- Car detection example
    - first create a level training set, so x and y with closely cropped example of cars
    - given labeled training set, train convnet to output y, 0 or 1 if car

- Sliding windows detection
    - input into conv net small rectangular region
    - move the window and run conv net
    - go through ever region of the size
    - then repeat with larger window
    - disadvantage is the computational cost
    - fortunately can be fixed convolutionally

Convolutional Implementation of Sliding Windows

- Turning FC layer into convolutional layers
    - view y as class probs of the softmax output

    - convolve last conv layer with 400 5x5 filters
    - same output size
    - then 400 1x1 filters
    - then 4 1x1 filters to replace the softmax output
- Convolution implementation of sliding windows
    - conv net input is slightly smaller than the image size
    - shift the window size four times to get four labels
    - the computation is highly duplicative
    - share the computation
    - the conv is set and take crops in strides of the input image

    - instead of four separate forward props, combines into one
    - the max pool is what corresponds to the stride of the window slide

    - problem is that the position of the bounding boxes won’t be too accurate

Bounding Box Predictions

- Output accurate bounding boxes
    - bounding box might not even be square
    - YOLO alg → you only look once
    - put down a grid
    - labels for training for each grid cell → y = [Pc, bx, by, bh, bw, c1, c2, c3]
    - find the prediction of the object’s bounding box in the grid cell that has objects
    - target output volume is 3x3x8, for example

    - works as long as only one object per grid cell
    - usually grid is larger, 19x19 for example
    - the model goes form input image 100x100x3, for example, through conv and maxpool … to 3x3x8, 3x3 being the size of the grid and 8 being the y vector
    - assign the midpoint of the object the the specific grid cell, even if it spans multiple grid cells
    - so you take the sliding convolution but implement per grid cell to be able to use the y vectors
    - pretty efficient
- Specify the bounding boxes
    - bx and by is between 0 and 1
    - bh and bw can be greater than 1

Intersection Over Union

- how to tell if object detection alg is working well
- Evaluating object localization
    - is the output bounding box a good one or a bad one
    - union is area contained in both bounding boxes
    - intersection is the intersection
    - compute size of intersection divided by size of union
    - by convention, correct if IoU ≥ 0.5, can choose other threshold

    - mapping localization to accuracy
    - measure of the overlap of two bounding boxes

Non-max Suppression

- one problem of the detection alg is that the model can detect an object multiple times
- Non-max suppression example
    - since every grid cell is running a object classification and localization alg, multiple cells might think it has the midpoint of the car, for example

    - multiple grid cells can think Pc is large

    - look at the probs, take the largest and highlight
    - then suppress the remaining boxes that have overlap
    - cycle through the remaining highest prob
- Non-max suppression alg
    - with a 19x19 grid, each output prediction is the vector [pc, bx, by, bh, bw]
    - discard all boxes with pc ≤ 0.6
    - while there are any remaining boxes
        - pick the box with the largest Pc, output that as a prediction
        - discard any remaining box with IoU ≥ 0.5 with the box output in the previous step
    - to detect three objects, for example, then the output vector will have three additional components
    - independently carry out non-max suppression three times

Anchor Boxes

- Overlapping objects
    - midpoint of two objects are almost identical
    - for the grid cell, y can’t output more than one object, c1, c2, c3, so it will have to pick
    - predefine two different shapes called anchor boxes
    - associate two different shapes and double the y vector
    - pc, bx, …, c2, c3 for anchor box 1 and pc, bx, …, c2, c3 for anchor box 2

- Anchor box alg
    - previously
        - each object in training image is assigned to grid cell that contains that object’s midpoint
    - with two anchor boxes
        - each object in training image is assigned to grid cell that contains object’s midpoint and anchor box for the grid cell with highest IoU
        - check which bounding box anchor box has the highest IoU and is assigned to a (grid cell, anchor box) pair
        - output goes from 3x3x8 to 3x3x16, for example
- Anchor box example
    - pedestrian is assigned to anchor box 1
    - car is more similar to anchor box 2
    - if only car, then the pedestrian part of the y vector will be don’t cares
    - what about three objects → this alg has no great way

    - rarely that two objects have the same midpoint
    - allows the learning alg to specialize better
    - how to choose anchor boxes → choose five to ten that spans a lot of shapes
    - use k-means alg that are common
    - choosing by hand is common

YOLO Algorithm

- Training
    - suppose alg is detecting three objects
        - pedestrian
        - car
        - motorcycle
    - if two anchor boxes then outputs will be 3x3 x 2x8, 8 is five plus the number of classes
    - in the example, the bounding box has a slightly higher IoU with anchor box 2, so the car gets associated with the lower portion of the vector, the other is don’t care
- Making predictions
    - if pc is zero, ignore
    - else, bounding box for the car
- Outputting the non-max suppressed outputs
    - for each grid cell, get two predicted bounding boxes
    - get rid of low probability predictions
    - for each class, use non-max suppression to generate final predictions

Region Proposals

- Region proposal: R-CNN
    - downside of yolo is that the alg is classifying a lot of regions where there is clearly no object
    - pick just a few regions or windows to run the cnn
    - runs a segmentation alg
    - get around 2000 blobs and run on just those
    - alg is slow
- Faster algorithms
    - R-CNN
        - propose regions
        - classify proposed regions one at a time
        - output label and bounding box
    - Fast R-CNN
        - propose regions
        - use convolution implementation of sliding windows to classify all the proposed regions
        - propose regions step is still quite slow
    - Faster R-CNN
        - use convolutional network to propose regions
        - still slower than yolo

Semantic Segmentation with U-Net

- draw a careful outline around the object
- Object Detection vs. Semantic Segmentation
    - boxes vs pixels

- Motivation for U-Net
    - helpful for doctors to know which pixels correspond to an area
    - tumor detection, saves time

- Per-pixel class labels
    - two class labels
    - 1 for car, 0 for not car
    - output 1 or 0 for every pixel
    - alternatively, car is one class, building is a second class, and the road is a third class

- Deep Learning for Semantic Segmentation
    - start with object recognition NN classifier
    - remove last few layers
    - the layers need to get bigger

Transpose Convolutions

- blowing up a 2x2 into a 4x4, for example
- Transpose Convolution
    - normal convolution
        - filters shrink the input image
    - transpose conv
        - smaller image convolved with a larger filter to get larger output
    - image size of 2x2, filter 3x3, padding 1, stride 2
    - interlapping values overlap → add

U-Net Architecture Intuition

- Deep Learning for Semantic Segmentation
    - normal convs followed by transpose convs
    - first part compresses the image
    - lost a lot of spatial information
    - second part turns the image back
    - add skip connections from earlier layers to later layers
    - why is because two types of information is useful
        - high level contextual information, but missing details spatial information
        - skip connection allow the NN to use the high res low level feature information

U-Net Architecture

- looks like the letter U

- first part is normal feed forward conv layers
- maxpool reduces height and width
- repeat x3
- trans conv added with skip connection
- repeat x3
- map to segmentation map with 1x1 conv
- output volume is h x w x n_classes
- every pixel has n_classes number of prop, take argmax to classify