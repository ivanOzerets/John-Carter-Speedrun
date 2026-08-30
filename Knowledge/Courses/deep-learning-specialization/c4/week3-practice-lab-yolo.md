Two YOLO papers

- [Redmon et al., 2016](https://arxiv.org/abs/1506.02640)
- [Redmon and Farhadi, 2016](https://arxiv.org/abs/1612.08242)

Packages

```python
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

%matplotlib inline
```

Problem Statement

- working on a self-driving car
- as a critical component of the project, first build a car detection system
- to collect data, a camera is mounded to the hood of the car which takes pictures of the road ahead every few seconds
- all the images are gathered into a folder and labelled by drawing bounding boxes around every car found
- an example of the bounding boxes

- if there are 80 classes the object detector should recognize, they can be represented with the class label c eighter as an integer from 1 to 80, or as an 80D vector one component of which is 1, and the rest 0
- because the YOLO model is very computationally expensive to train, the pre-trained weights are already loaded

YOLO

- popular alg because it achieves high accuracy while also being able to run in real time
- the alg “only looks once” at the image in the sense that it requires only one forward prop pass though the network to make predictions
- after non-max suppression, it then outputs recognized objects together with bounding boxes
- Model Details
    - inputs and outputs
        - the input is a batch of images, and each image has the shape (608,608,3)
        - the output is a list of bounding boxes along with the recognized classes
        - each bounding box is represented by six numbers (pc, bx, by, bh, bw, c)
        - if you expand c into an 80D vector, each bounding box is then represented by 85 numbers
    - anchor boxes
        - anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes
        - five anchor boxes were chosen to cover 80 classes in this assignment
        - the dimension of the encoding tensor of the second to last dimension based ont eh anchor boxes is (m, nh, nw, anchors, classes)
        - the yolo architecture is: IMAGE (m, 608, 608, 3) → DEEP CNN → ENCODING (m, 19, 19, 5, 85)
    - encoding
        - greater detail

        - if the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object
        - since five anchor boxes are being used, each of the 19x19 cells thus encodes information about five boxes
        - anchor boxes are defined only by their width and height
        - for simplicity, flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425)

    - class score
        - for each box (of each cell), compute the following element-wise product and extract a probability that the box contains a certain class
        - the class score is score_c,i = pc x ci: the probability that there is an object pc times the prob that the object is a certain class ci

        - for box 1, the probability that an object exists is p1 = 0.6
        - so there’s a 60 percent chance that an object exists in box 1
        - the probability that the object is the class “category 3” is c3 = 0.73
        - the score for box 1 and for category 3 is score_1,3 = .6 x 0.73 = 0.44
        - calculate the score for all 80 classes in box 1, and find that the score for the car class is the max
        - assign the score 0.44 and class “3” to this box “1”
    - visualizing classes
        - one way to visualize what yolo is predicting on an image
            - for each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one max for each of the five anchor boxes
            - color that grid cell according to what object that grid cell considers the most likely
        - results in a picture

    - visualizing bounding boxes
        - another way to visualize yolo’s output is to plot the bounding boxes that it outputs

    - non-max suppression
        - the only boxes plotted are ones for which the model had assigned a high probability, but this is still too many boxes
        - reduce the alg’s output to a much smaller number of detected objects
        - to do so, use non-max suppression
            - get rid of boxes with a low score, meaning, the box is not very confident about detecting a class, either due to the low probability of an object, or low prob of the particular class
            - select only one box when several boxes overlap with each other and detect the same object
- Filtering with a Threshold on Class Scores
    - first apply a filter by thresholding, meaning to get rid of any box for which the class ‘score’ is less than a chosen threshold
    - the model gives a total of 19x19x5x85 numbers, with each box described by 85 numbers
    - its convenient to rearrange to 19x19x425 dimensional tensor for the following variables
        - box_confidence: tensor of shape (19, 19, 5, 1) containing pc (confidence probability that there’s some object) for each of the five boxes predicted in each of the 19x19 cells
        - boxes: tensor of shape (19, 19, 5, 4) containing the midpoint and dimensions (bx, by, bh, bw) for each of the five boxes in each cell
        - box_class_probs: tensor of shape (19, 19, 5, 80) containing the ‘class probabilities’ (c1, c2, …, c80) for each of the 80 classes for each of the five boxes per cell
    - Exercise 1 - yolo_filter_boxes
        - implement yolo_filter_boxer()
        - compute box scores by doing the elementwise product

            ```python
            a = np.random.randn(19, 19, 5, 1)
            b = np.random.randn(19, 19, 5, 80)
            c = a * b # shape of c will be (19, 19, 5, 80)
            ```

        - for each box find
            - the index of the class with the max box score
            - the corresponding box score
            - tf.math.argmax
            - tf.math.reduce_max
            - for the axis param of argmax and reduce_max, to select the last axis, use axis = -1, similar to python array indexing where to select the last position of an array use arrayname[-1]
            - applying reduce_max normally collapses the axis for which the max is applied
            - keepdims=Flale s the default option, and allows that dimension to be removed
            - don’t need to keep the last dimension after applying the max here
        - create a mask by using a threshold
        - ([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) returns [false, true, false, false, true]
        - the mask should be true for the boxes to keep
        - use tf to apply the mask to box_class_scores, boxes, and box_classes to filter out the boxes not wanted
        - left with just the subset of boxes to keep
        - tf.boolean_mask, keep default axis=None

            ```python
            # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
            # GRADED FUNCTION: yolo_filter_boxes

            def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
                """Filters YOLO boxes by thresholding on object and class confidence.

                Arguments:
                    boxes -- tensor of shape (19, 19, 5, 4)
                    box_confidence -- tensor of shape (19, 19, 5, 1)
                    box_class_probs -- tensor of shape (19, 19, 5, 80)
                    threshold -- real value, if [ highest class probability score < threshold],
                                 then get rid of the corresponding box

                Returns:
                    scores -- tensor of shape (None,), containing the class probability score for selected boxes
                    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
                    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

                Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
                For example, the actual output size of scores would be (10,) if there are 10 boxes.
                """

                ### START CODE HERE
                # Step 1: Compute box scores
                ##(≈ 1 line)
                box_scores = box_confidence * box_class_probs

                # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
                ##(≈ 2 lines)
                # IMPORTANT: set axis to -1
                box_classes = tf.math.argmax(box_scores, axis=-1)
                box_class_scores = tf.math.reduce_max(box_scores, axis=-1)

                # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
                # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
                ## (≈ 1 line)
                filtering_mask = (box_class_scores >= threshold)

                # Step 4: Apply the mask to box_class_scores, boxes and box_classes
                ## (≈ 3 lines)
                scores = tf.boolean_mask(box_class_scores, filtering_mask)
                boxes = tf.boolean_mask(boxes, filtering_mask)
                classes = tf.boolean_mask(box_classes, filtering_mask)
                ### END CODE HERE

                return scores, boxes, classes

            '''
            scores[2] = 9.270486
            boxes[2] = [ 4.6399336  3.2303846  4.431282  -2.202031 ]
            classes[2] = 8
            scores.shape = (1789,)
            boxes.shape = (1789, 4)
            classes.shape = (1789,)
            '''
            ```

        - in the test for yolo_filter_boxes, use random numbers to test the function
        - in real data, the box_class_probs would contain non-zero values between 0 and 1 for the probabilities
        - the box coordinates in boxes would also be chosen so that lengths and weights are non-negative
- Non-max Suppression
    - even after filtering by thresholding over the class scores, still end up with a lot of overlapping boxes

    - non-max suppression uses the very important funciton called “intersection over Union”, or IoU

    - Exercise 1 - iou
        - implement iou()
        - the code uses the convention that (0,0) is the top-left corner of the image, (1,0) is the upper-right corner ..
        - for this exercise, a box is defined using its two corners: upper left and lower right, instead of using the midpoint, height and width,
        - this makes it a bit easier to calculate the intersection
        - to calculate the area of a rectangle, multiply its height (y2 - y1), by its width (x2 - x1)
        - since (x1, y1) is the top left and (x2, y2) are the bottom right, these differences should be non-negative
        - to find the intersection of the two boxes (xi1, yi1, xi2, yi2)
            - draw some example of paper to clarify the concept
            - the top left corner of the intersection (xi1, yi1) is found by comparing the top left corners (x1, y1) of the two boxes and finding a vertex that has an x-coordinate that is closer to the right, and y-coordinate that is closer to the bottom
            - the bottom right corner of the intersection (xi2, yi2) is found by comparing the bottom right corners (x2, y2) of the two boxes and finding a vertex whose x-coordinate is closer to the left, and the y-coordinate that is closer to the top
            - the two boxes may have no intersection
            - to detect this, the intersection coordinates calculated end up being the top right and/or bottom left corners of an intersection box
            - another way to think of this is if at least on of height or width is non-negative, then there is no intersection (intersection area is zero)
            - the two boxes may intersect at the edges or vertices, in which case the intersection area is still zero
            - this happens when either the height or width (or both) of the calculated intersection is zero
        - xi1 = maximum of the x1 coordinates of the two boxes
        - yi1 = maximum of the y1 coordinates
        - xi2 = minimum of the x2 coords
        - yi2 = min of the y2 coords
        - inter_area = use max(height, 0) and max(width, 0)

            ```python
            # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
            # GRADED FUNCTION: iou

            def iou(box1, box2):
                """Implement the intersection over union (IoU) between box1 and box2

                Arguments:
                box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
                box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
                """

                (box1_x1, box1_y1, box1_x2, box1_y2) = box1
                (box2_x1, box2_y1, box2_x2, box2_y2) = box2

                ### START CODE HERE
                # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
                ##(≈ 7 lines)
                xi1 = max(box1_x1, box2_x1)
                yi1 = max(box1_y1, box2_y1)
                xi2 = min(box1_x2, box2_x2)
                yi2 = min(box1_y2, box2_y2)
                inter_width = xi2 - xi1
                inter_height = yi2 - yi1
                inter_area = max(inter_width, 0) * max(inter_height, 0)

                # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
                ## (≈ 3 lines)
                box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
                box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
                union_area = box1_area + box2_area - inter_area
                print (union_area, inter_area)

                # compute the IoU
                iou = inter_area / union_area
                ### END CODE HERE

                return iou

            '''
            iou for intersecting boxes = 0.14285714285714285
            iou for non-intersecting boxes = 0.0
            iou for boxes that only touch at vertices = 0.0
            iou for boxes that only touch at edges = 0.0
            '''
            ```

- YOLO Non-max Suppression
    - implement non-max suppression
        - select the box with the highest score
        - compute the overlap of this box with all other boxes of the same class
        - remove any boxes that significantly overlap (IoU ≥ iou_threshold)
        - repeat the process: go back to stop 1 and iterate until no remaining boxes have a score lower than the currently selected box
    - this process effectively eliminates boxes that overlap significantly with the selected boxes, leaving only the best candidates
    - for implementation, consider using a divide-and-conquer approach
        - divide all the boxes by class
        - apply non-max suppression for each class individually
        - finally, merge the results from each class into a single set of boxes
    - Exercise 3 - yolo_non_max_suppression
        - implement yolo_non_max_suppression() using tf
        - tf has two built-in functions that are used to implement non-max suppression
        - reference documentation
            - tf.image.non_max_suppression()

                ```python
                tf.image.non_max_suppression(
                  boxes, scores, max_output_size, iou_threshold=0.5,
                  score_threshold=float('-inf'), name=None
                )
                ```

            - for tf.image.non_max_suppression(), only set boxes, scores, max_output_size and iou_threshold parameters
            - tf.gather()

                ```python
                tf.gather(
                  params, indices, validate_indices=None, axis=None, batch_dims=0, name=None
                )
                ```

            - for if.gather(), only set params and indices params
            - tf.boolean_mask()

                ```python
                tf.boolean_mask(
                  tensor, mask, axis=None, name='boolean_mask'
                )
                ```

            - for tf.boolean_mask(), only set tensor and mask params
            - tf.concat()

                ```python
                tf.concat(
                  values, axis, name='concat'
                )
                ```

            ```python
            # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
            # GRADED FUNCTION: yolo_non_max_suppression

            def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
                """
                Applies Non-max suppression (NMS) to set of boxes

                Arguments:
                scores -- tensor of shape (None,), output of yolo_filter_boxes()
                boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
                classes -- tensor of shape (None,), output of yolo_filter_boxes()
                max_boxes -- integer, maximum number of predicted boxes you'd like
                iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

                Returns:
                scores -- tensor of shape (None, ), predicted score for each box
                boxes -- tensor of shape (None, 4), predicted box coordinates
                classes -- tensor of shape (None, ), predicted class for each box

                Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
                function will transpose the shapes of scores, boxes, classes. This is made for convenience.
                """
                boxes = tf.cast(boxes, dtype=tf.float32)
                scores = tf.cast(scores, dtype=tf.float32)

                nms_indices = []
                classes_labels = tf.unique(classes)[0] # Get unique classes

                for label in classes_labels:
                    filtering_mask = classes == label

                #### START CODE HERE

                    # Get boxes for this class
                    # Use tf.boolean_mask() with 'boxes' and `filtering_mask`
                    boxes_label = tf.boolean_mask(boxes, filtering_mask)

                    # Get scores for this class
                    # Use tf.boolean_mask() with 'scores' and `filtering_mask`
                    scores_label = tf.boolean_mask(scores, filtering_mask)

                    if tf.shape(scores_label)[0] > 0:  # Check if there are any boxes to process

                        # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
                        ##(≈ 5 lines)
                        nms_indices_label = tf.image.non_max_suppression(
                                boxes_label,
                                scores_label,
                                max_boxes,
                                iou_threshold) 

                        # Get original indices of the selected boxes
                        selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)

                        # Append the resulting boxes into the partial result
                        # Use tf.gather() with 'selected_indices' and `nms_indices_label`
                        nms_indices.append(tf.gather(selected_indices, nms_indices_label))

                # Flatten the list of indices and concatenate
                # Use tf.concat() with 'nms_indices' and `axis=0`
                nms_indices = tf.concat(nms_indices, axis=0)

                # Use tf.gather() to select only nms_indices from scores, boxes and classes
                ##(≈ 3 lines)
                scores = tf.gather(scores, nms_indices)
                boxes = tf.gather(boxes, nms_indices)
                classes = tf.gather(classes, nms_indices)

                ### END CODE HERE       

                # Sort by scores and return the top max_boxes
                sort_order = tf.argsort(scores, direction='DESCENDING').numpy()
                scores = tf.gather(scores, sort_order[0:max_boxes])
                boxes = tf.gather(boxes, sort_order[0:max_boxes])
                classes = tf.gather(classes, sort_order[0:max_boxes])

                return scores, boxes, classes

            '''
            iou:    	0.22780233025311375
            scores:  	[0.855]
            boxes:  	[[0.45 0.2  1.01 2.6 ]]
            classes:	[0]
            '''
            ```

- Wrapping Up the Filtering
    - implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes
    - Exercise 4 - yolo_eval
        - implement yolo_eval() which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS
        - there’s a few ways of representing boxes, such as via their corners or via their midpoint and height/width
        - TOLO converts between a few such formats at different times

            ```python
            boxes = yolo_boxes_to_corners(box_xy, box_wh) 
            ```

        - converts the yolo box coordinates (x, y, w, h) to box corner’s coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes

            ```python
            boxes = scale_boxes(boxes, image_shape)
            ```

        - YOLO’s network was trained to run on 608x608 images
        - testing the data on a different size image, for example, the car detection dataset had 720x1280 images
        - this step rescales the boxes so that they can be plotted on top of the original 720x1280 image

            ```python
            def yolo_boxes_to_corners(box_xy, box_wh):
                """Convert YOLO box predictions to bounding box corners."""
                box_mins = box_xy - (box_wh / 2.)
                box_maxes = box_xy + (box_wh / 2.)

                return tf.keras.backend.concatenate([
                    box_mins[..., 1:2],  # y_min
                    box_mins[..., 0:1],  # x_min
                    box_maxes[..., 1:2],  # y_max
                    box_maxes[..., 0:1]  # x_max
                ])

            ```

        ```python
        # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION: yolo_eval

        def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
            """
            Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

            Arguments:
            yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                            box_xy: tensor of shape (None, 19, 19, 5, 2)
                            box_wh: tensor of shape (None, 19, 19, 5, 2)
                            box_confidence: tensor of shape (None, 19, 19, 5, 1)
                            box_class_probs: tensor of shape (None, 19, 19, 5, 80)
            image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
            max_boxes -- integer, maximum number of predicted boxes you'd like
            score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
            iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

            Returns:
            scores -- tensor of shape (None, ), predicted score for each box
            boxes -- tensor of shape (None, 4), predicted box coordinates
            classes -- tensor of shape (None,), predicted class for each box
            """

            ### START CODE HERE

            # Retrieve outputs of the YOLO model (≈1 line)
            box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

            # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
            boxes = yolo_boxes_to_corners(box_xy, box_wh)

            # Use the function `yolo_filter_boxes` you've implemented to perform Score-filtering with a threshold of score_threshold
            scores, boxes, classes = yolo_filter_boxes(boxes, # Use boxes
                                          box_confidence, # Use box confidence
                                          box_class_probs, # Use box class probability
                                          score_threshold  # Use threshold=score_threshold
                                         )

            # Scale boxes back to original image shape.
            boxes = scale_boxes(boxes, image_shape)

            # Use the function `yolo_non_max_suppression` you've implemented to perform Non-max suppression with 
            # maximum number of boxes set to max_boxes and a threshold of iou_threshold
            scores, boxes, classes = yolo_non_max_suppression(scores, # Use scores
                                          boxes, # Use boxes
                                          classes, # Use classes
                                          max_boxes, # Use max boxes
                                          iou_threshold  # Use iou_threshold=iou_threshold
                                         )

            ### END CODE HERE

            return scores, boxes, classes

        '''
        scores[2] = 171.60194
        boxes[2] = [-1240.3483 -3212.5881  -645.78    2024.3052]
        classes[2] = 16
        scores.shape = (10,)
        boxes.shape = (10, 4)
        '''
        ```

Test YOLO Pre-Trained Model on Images

- use a pre-trained model and test it on the car detection dataset
- Defining Classes, Anchors and Image Shape
    - trying to detect 80 classes, and using five anchor boxes
    - the information on the 80 classes and five boxes is gathered in two files
    - coco_classes.txt and yolo_anchors.txt
    - read class names and anchors from text files
    - the car detection dataset has 720x1280 images, which are preprocessed into 608x608 images

        ```python
        class_names = read_classes("model_data/coco_classes.txt")
        anchors = read_anchors("model_data/yolo_anchors.txt")
        model_image_size = (608, 608) # Same as yolo_model input layer size
        ```

- Loading a Pre-trained Model
    - training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes
    - load an existing pre-trained Keras YOLO model stored in ‘yolo.h5’
    - these weights come from the official YOLO website, and were converted using a function written by allan zelener
    - these are parameters from teh yolov2 model

        ```python
        yolo_model = load_model("model_data/", compile=False)
        ```

    - this loads the weights of the trained yolo model

        ```python
        yolo_model.summary()

        '''
        Model: "functional_1"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_1 (InputLayer)            [(None, 608, 608, 3) 0                                            
        __________________________________________________________________________________________________
        conv2d (Conv2D)                 (None, 608, 608, 32) 864         input_1[0][0]                    
        __________________________________________________________________________________________________
        ...
        __________________________________________________________________________________________________
        conv2d_21 (Conv2D)              (None, 19, 19, 1024) 11796480    concatenate[0][0]                
        __________________________________________________________________________________________________
        batch_normalization_21 (BatchNo (None, 19, 19, 1024) 4096        conv2d_21[0][0]                  
        __________________________________________________________________________________________________
        leaky_re_lu_21 (LeakyReLU)      (None, 19, 19, 1024) 0           batch_normalization_21[0][0]     
        __________________________________________________________________________________________________
        conv2d_22 (Conv2D)              (None, 19, 19, 425)  435625      leaky_re_lu_21[0][0]             
        ==================================================================================================
        Total params: 50,983,561
        Trainable params: 50,962,889
        Non-trainable params: 20,672
        '''
        ```

    - this model coverts a preprocessed batch of input images into a tensor of shape (m, 19, 19, 5, 85
- Convert Output of the Model to Usable Bounding Box Tensors
    - the output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion
    - need to call yolo_head to format the encoding of the model from yolo_model into something decipherable
    - yolo_model_outputs = yolo_model(image)data)
    - yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    - the variable yolo_outputs will be defined as a set of four tensors that can then be used as input by the yolo_eval function
- Filtering Boxes
    - yolo_outputs give all the predicted boxes of yolo_model in the correct format
    - to perform filtering and select only the best boxes, call yolo_eval

        ```python
        out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)
        ```

- Run the YOLO on an image
    - create a graph that can be summarized as follows
        - yolo_model.input is given to yolo_model
        - the model is used to compute the output yolo_model.output
        - you_model.output is processed by yolo_head
        - it gives yolo_outputs
        - yolo_outputs goes through a filtering function, yolo_eval
        - it outputs the predictions: out_scores, out_boxes, out_classes
    - predict(image_file) runs teh graph to test YOLO on an image to compute out_scores, out_boxes, out_classes

        ```python
        image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
        ```

    - which opens the image file and scales, reshapes and normalizes the image and returns

        ```python
        image: a python (PIL) representation of your image used for drawing boxes. You won't need to use it.
        image_data: a numpy-array representing the image. This will be the input to the CNN.
        ```

        ```python
        def predict(image_file):
            """
            Runs the graph to predict boxes for "image_file". Prints and plots the predictions.

            Arguments:
            image_file -- name of an image stored in the "images" folder.

            Returns:
            out_scores -- tensor of shape (None, ), scores of the predicted boxes
            out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
            out_classes -- tensor of shape (None, ), class index of the predicted boxes

            Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
            """

            # Preprocess your image
            image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

            yolo_model_outputs = yolo_model(image_data)
            yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

            out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

            # Print predictions info
            print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
            # Generate colors for drawing bounding boxes.
            colors = get_colors_for_classes(len(class_names))
            # Draw bounding boxes on the image file
            #draw_boxes2(image, out_scores, out_boxes, out_classes, class_names, colors, image_shape)
            draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
            # Save the predicted bounding box on the image
            image.save(os.path.join("out", image_file), quality=100)
            # Display the results in the notebook
            output_image = Image.open(os.path.join("out", image_file))
            imshow(output_image)

            return out_scores, out_boxes, out_classes
        ```

        ```python
        out_scores, out_boxes, out_classes = predict("test.jpg")

        '''
        Found 10 boxes for images/test.jpg
        car 0.89 (367, 300) (745, 648)
        car 0.80 (761, 282) (942, 412)
        car 0.74 (159, 303) (346, 440)
        car 0.70 (947, 324) (1280, 705)
        bus 0.67 (5, 266) (220, 407)
        car 0.66 (706, 279) (786, 350)
        car 0.60 (925, 285) (1045, 374)
        car 0.44 (336, 296) (378, 335)
        car 0.37 (965, 273) (1022, 292)
        traffic light 0.36 (681, 195) (692, 214)
        '''
        ```

    - the model was able to detect 80 different classes

Summary for YOLO

- input image (608, 608, 3)
- the input image goes through a CNN, resulting in a (19,19,5,85) dimensional output
- after flattening the last two dimensions, the output is a volume of shape (19, 19, 425)
    - each cell in a 19x19 grid over the input image gives 425 numbers
    - 425 = 5x85 because each cell contains predictions for five boxes, corresponding to five anchor boxes
    - 85 = 5 + 80 where 5 is because (pc, bx, by, bh, bw) has five numbers, and 80 is the number of classes to detect
- select only few boxes based on
    - score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - non-max suppression: compute the intersection over union and avoid selecting overlapping boxes
- gives the final output