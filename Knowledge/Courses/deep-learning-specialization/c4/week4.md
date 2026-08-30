What is Face Recognition?

- Face recognition
    - liveness detection
- Face verification vs face recognition
    - verification
        - input image, name/ID
        - output whether the input image is that of the claimed person
        - 1 : 1 problem
    - recognition
        - much harder
        - has a database of K persons
        - get an input image
        - output ID if the image is any of the K persons (or ‘not recognized’)
        - if verification is 99%, and the database has 100 persons, then the verification process would have to increase to 99.9% to maintain a good accuracy on recognition

One Shot Learning

- challenge of face recognition
- recognize a person give just one example
- deep learning alg don’t work well with just on training example
- learning from just one example
- also, if more persons get added, now the output changes
- Learning a ‘similarity’ function
    - d(img1, img2) = degree of difference between images
    - if d(img1, img2) ≤ tau, same, > tau, different

Siamese Network

- the last list of 128 numbers, f(x1), of the fc layers, is thought of as an encoding of the input image x1
- a new image has a new encoding f(x2)
- if the encodings are good representations of the images, then define image d(x1, x2) = ||f(x1) - f(x2)||^2
- came from system called DeepFace
- Goal of learning
    - parameters of NN define an encoding f(xi)
    - learn params so that:
        - if xi, xj are the same person, ||f(xi) - f(xj)||^2 is small
        - if xi, xj are different persons, ||f(xi) - f(xj)||^2 is large

Triplet Loss

- Learning Objective
    - look at anchor image
    - positive image distance to be small
    - negative image distance to be larger
    - always looking at three images
    - A, P, and N
    - want ||f(A) - f(P||^2 ≤ ||f(A) - f(N)||^2
    - or d(A, P) ≤ d(A, N)
    - ||f(A) - f(P||^2 - ||f(A) - f(N)||^2 ≤ 0
    - a trivial way to satisfy the expression is make everything zero
    - to make sure the NN doesn’t just output all zeros, modify the expression with an alpha margin

    - margin pushes the anchor positive and anchor negative pair further away from each other
- Loss function
    - given 3 images, A, P, N

    - cost function

    - training set: 10k pictures of 1k persons
    - to train, you do need multiple pictures of the same person
- Choosing the triplets A, P, N
    - during training, if A, P, N are chosen randomly, d(A, P) + alpha ≤ d(A, N)
    - chances are that A and N are much different than A and P, so easily satisfied
    - so won’t learn much
    - need to chose triplets that are hard to train on
- Training set using triplet loss
    - use GD to minimize J

Face Verification and Binary Classification

- face classification can be also posed as a straight binary classification problem
- Learning the similarity function
    - could create two models where the outputs are fed into a logistic regression unit to make a prediction

    - rather than just feeding in the encodings, take difference between the encodings

    - ^ wi is actually wk
    - other variations

    - siamese means the parameters of the model are the same
    - can precompute encodings of database images to save computation
- Face verification supervised learning
    - now just pairs of images
    - pairs are same or different
    - pairs used to train the model

What is Neural Style Transfer

- Neural style transfer
    - contact (C), style (S), and generated image (G)

    - look at extracted features at various layers of the conv net

What are deep ConvNets learning?

- Visualizing what a deep network is learning
    - pick a unit in layer 1

    - find the nine image patches that maximize the unit’s activation
    - hidden layer unit, especially in early layers, will only see a small portion of the image
    - repeat for other units

    - simple features such as an edge or a particular color
- Visualizing deep layers
    - the deeper, the more of the image is seen per units
    - layer 2

    - layer 3

    - layer 4

    - layer 5

Cost Function

- Neural style transfer cost function
    - J(G) measure how good is the generated image
    - Jcontent(C, G) how similar is the content of the generated image to the generated image G
    - Jstyle(S, G) how similar is the content of the style image to the generated image G
    - then add hypers alpha and beta

    - two hypers seem redundant, however the original paper uses two
- Find the generated image G
    - initiate G randomly
        - 100x100x3, for example
    - use gradient descent to minimize J(G)
    - updating pixel values

Content Cost Function

- say using hidden layer l to compute content cost
- if l is small, forces generated image to have pixel values similar to content image
- if l is large, then the representations can be too loose
- dog identifier layer, just have a dog somewhere in the image
- so somewhere in between for layer l
- use pre-trained ConvNet (VGG network)
- let a[l](C) and a[l][G] be the activation of layer l on the images
- if a[l](C) and a[l][G] are similar, both images have similar content

- 1/2 is just normalization which is not very important

Style Cost Function

- Meaning of the ‘style’ of an image
    - say using layer l’s activation to measure style
    - define style as correlation between activations across channels

    - how correlated are the activations across different channels

- Intuition about style of an image
    - for two channels to be highly correlated, for example, one channel detecting vertical lines and another detecting orange tint, a give image section will spike the activations when it is orange tinted and has vertical stripes

    - uncorrelated mean that vertical lines don’t have vertical tints to it
    - comparing channels between G and S
- Style matrix
    - compute a matrix
    - let a^[l]_i,j,k = activation at (i, j, k)
    - ‘correlation’ is actual unnormalized cross-covariance
    - matrix for both style matrix and generated image

    - ‘gram matrix’ known in lin alg

- Style cost function
    - get more visually pleasing results if style cost function from multiple layers is used

    - overall cost function

1D and 3D Generalizations

- many idea apply to 1D and 3D data
- Convolutions in 2D and 1D
    - 1D example is EKG results

- 3D data
    - CT scan
    - slices of the body
- 3D convolution
    - just add an extra dimension

    - another use could be movie data

References

### **Week 1:**

- [The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model) (TensorFlow Documentation)
- [The Functional API](https://www.tensorflow.org/guide/keras/functional) (TensorFlow Documentation)

### **Week 2:**

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren & Sun, 2015)
- [deep-learning-models/resnet50.py/](https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py) (GitHub: fchollet)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, & Adam, 2017)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (Sandler, Howard, Zhu, Zhmoginov &Chen, 2018)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)

### **Week 3:**

- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (Redmon, Divvala, Girshick & Farhadi, 2015)
- [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (Redmon & Farhadi, 2016)
- [YAD2K](https://github.com/allanzelener/YAD2K) (GitHub: allanzelener)
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Fully Convolutional Architectures for Multi-Class Segmentation in Chest Radiographs](https://arxiv.org/abs/1701.08816) (Novikov, Lenis, Major, Hladůvka, Wimmer & Bühler, 2017)
- [Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks](https://arxiv.org/abs/1705.03820) (Dong, Yang, Liu, Mo & Guo, 2017)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger, Fischer & Brox, 2015)

### **Week 4:**

- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf) (Schroff, Kalenichenko & Philbin, 2015)
- [DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) (Taigman, Yang, Ranzato & Wolf)
- [facenet](https://github.com/davidsandberg/facenet) (GitHub: davidsandberg)
- [How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) (Jason Brownlee, 2019)
- [keras-facenet/notebook/tf_to_keras.ipynb](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb) (GitHub: nyoki-mtl)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys, Ecker & Bethge, 2015)
- [Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
- [TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
- [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) (Simonyan & Zisserman, 2015)
- [Pretrained models](https://www.vlfeat.org/matconvnet/pretrained/) (MatConvNet)