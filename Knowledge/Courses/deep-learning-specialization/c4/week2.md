Why look at case studies?

- how to put together building blocks to form effective convolutional NN
- intuition from seeing examples
- classic networks
    - LeNet-5
    - AlexNet
    - VGG
- ResNet
    - 152 layer network
- Inception

Classic Networks

- LeNet-5
    - start with image
    - goal to recognize hand written grayscale images
    - 6 filters with stride of one
    - average pool (now use max likely)
    - conv layer 16 filers
    - avg pool
    - fully connected 120 units
    - fully connected 84 units
    - yhat prediction

    - around 60k params
    - small compared to the millions of parameters networks of today
    - height and width tend to go down, whereas the channels go up
    - one or more conv followed by pools, repeat, fc, fc, output
    - sigmoid/tanh non-linearities were used back then
    - had non-linearity after pooling
- AlexNet
    - inputs 227x227x3 images
    - 96 filters, stride 4
    - max pool stride 2
    - 5x5 same conv
    - max pool stride 2
    - 3x3 same
    - 3x3 same
    - 3x3 same
    - max pool stride 2
    - fc 9216
    - fc 4096
    - fc 5096
    - about 60 million parameters

    - used ReLU
    - Local Response Normalization (not used much)
        - looks at one position across channels and normalized
    - easier to read
- VGG-16
    - simpler network
    - focus on conv 3x3 and same padding
    - max-pool 2x2 stride of 2
    - 224x224x3 image
    - conv 64 x2
    - pool
    - …

    - huge amount of parameters, even by today’s standard, 138 M
    - 16 refers to 16 layers with weights
    - simplicity made the model appealing
    - doubling filters is also a standard that was developed
    - VGG 19 is even bigger
    - systematic how filters go up by x2 and pooling goes down by x2

ResNets

- very deep NN are hard to train because of vanishing and exploding gradient problems
- feeding layer output way further in the network
- Residual block
    - main path: a[l] → linear → ReLU → linear → ReLU
    - shortcut: a[l] goes straight to the last non-linearity

    - also known as a skip connection
    - allows for deep NN
- Residual Network
    - plain network

    - as layers increase, training error increase
    - in theory it shouldn’t, but in reality it does get worse
    - not with ResNet

Why ResNets Work?

- Why do residual networks work?
    - assuming ReLU, so all non negative quantities
    - identity function is easy Residual blocks to learn
    - because z[l+2] can be canceled by W and b and are left with the original a[l]

    - so adding additional residual layers doesn’t hurt the model’s ability to do as well as the simpler network

    - if the additional hidden units learned something useful, then the model can do even better than just learning the identity function
    - in very deep nets without skip connections, the model has a hard time learning parameters for even the identity function
    - resnets work because its easy for the model to learn the identity function → getting out just the original input
    - assuming z[l+2] and a[l] are the same shape so resents have a lot of same convolutions
    - if different dimensional output required, add an extra matrix Ws, which could be a fixed matrix that implements zero padding or parameters to be learned
- ResNet
    - Plain Network vs ResNet
    - pooling layers need to make an adjustment to the shape with Ws

Networks in Networks and 1x1 Convolutions

- What does a 1x1 convolution do?
    - just multiplying by some number with one channel
    - but with multiple channels, each of the positions will have a element-wise product with all channel numbers with a non-linearity
    - results in a single number
    - as if taking, as input, the same position, all channels, and multiplying each them by that many channel weights and applying a non-linearity

    - basically having a fc between channel slice and a filter
    - sometimes called a network, or network idea
- Using 1x1 convolutions
    - to shrink height and width → pooling
    - to shrink channels → 32 1x1 filters

Inception Network Motivation

- size of filter of pooling? why not do them all
- Motivation for inception network
    - stack all the possible convs and pooling at the same time
    - pooling uses padding and stride of 1, unusual

- The problem of computational cost
    - number of multiplications for just the 5x5 conv
    - output volume x filter size

    - 1x1 can reduce computational cost by a factor of 10
- Using 1x1 convolution
    - first reduce channels with 1x1 conv
    - then 5x5 conv
    - known as a bottleneck layer

    - shrinking down with 1x1 conv, within reason, does not seem to hurt performance

Inception Network

- Inception module
    - use 1x1 before larger conv filter layers
    - maxpool is ‘same’ followed by 1x1 to shrink channels

    - give a 28x28x256
    - one inception module
- Inception network
    - puts blocks together
    - side branches take a hidden layer and predict at different levels
    - helps ensure that the features computed, even in hidden layers, are not too bad for predicting the output class of the image
    - appears to apply a regularization effect

    - developed at google
    - called googleNet

MobilNet

- another conv NN architecture used for comp vision
- allows to build and deploy new networks that work in low compute environments like mobile phones
- Motivation for MobilNets
    - low computational cost at deployment
    - useful for mobile and embedded vision applications
    - key idea: normal vs. depthwise-separable convolutions
- Normal Convolution
    - computational cost = number of filter params x number of positions x number of filters

- Depthwise Separable Convolution
    - two steps
    - depthwise conv followed by pointwise

- Depthwise Convolution
    - filters are fxf, one channel
    - you have nc of those filters
    - each filter is applied to one input channel

    - output is the same number of channels as input
    - number of computations go down by the channel size in this step
- Pointwise Convolution
    - take the intermediate output and convolve with nc’ 1x1xnc filters

    - number of computations is reduced because of 1x1 conv use
- Cost Summary
    - cost of normal convolution = 2160
    - cost of depthwise separable convolution = 432 + 240 = 672
    - about 3x savings
    - ration of the cost in general is 1/nc’ + 1/f^2
    - becomes actually around ten times cheaper

MobileNet Architecture

- idea is use a much less expensive depthwise convolution arch instead of an expensive conv layers
- the original paper used 13 layer of the depthwise
- then followed by pool, fc, softmax
- another improvement: MobileNet v2
- addition of a residual connection / skip connection
- also adds an expansion layer
- projection is called a pointwise convolution
- 17 layers this time

- the red box is called the bottleneck block
- MobileNet v2 Bottleneck
    - expansion 1x1 filter, but a lot of filters
    - expansion of factor of 6 is common
    - then depthwise conv
    - then apply a pointwise conv
    - projection because ‘projecting down’ to a smaller channel size

    - bottleneck block accomplishes two things
        - expansion increases size of representation, allows to have a richer function
        - because of memory constraint, pointwise shrinks it back down

EfficientNet

- tune MobileNet or other to a specific device?
- scale up or down
- to scale up
    - resolution
    - depth
    - width
    - compound scaling (some or all at one)
- what’s the rate? what’s the best scale up
- finds good trade off between r d and w

Using Open-Source Implementation

- often finicky to replicate papers because of a lot of details
- tuning hypers for example
- look online for an open source implementation
- git clone the url

Transfer Learning

- download weights
- ImageNet
- MS COCO
- Pascal types of data sets
- Smaller training set
    - with a small personal training set, take ImageNet, for example, weights and code, rewrite the softmax layer and freeze (don’t run GD on) the parameters and only train the personal softmax layer
    - some models even have trainableParameter or freeze
    - could precompute frozen activation from input to last output and save to disk
    - since frozen → its just a feature vector
- Medium size training set
    - could freeze fewer layers, just the beginning layers
- Large size training set
    - whole thing is used as initialization
    - then train the whole thing

Data Augmentation

- more data almost always helps
- Common augmentation method
    - mirroring horizontally
    - random cropping
    - rotation
    - shearing
    - local warping
- Color shifting
    - add to the r, g, and b channels different distortions
    - draw from a small distribution

    - different ways to sample color distortion
    - PCA → details in AlexNet paper, “PCA color augmentation”
    - if tinted, add and sub a lot from prominent channels
- Implementing distortions during training
    - data is usually stored on a harddisk
    - on a small amount of data, almost anything is okay
    - on a large training set, CPU thread loading images to implement distortions
    - forms a mini-batch of data and fed into training
    - both can run in parallel
    - DA has its own hypers

State of Computer Vision

- Data vs hand-engineering
    - most ML problems fall on spectrum of little data to a lot of data
    - image detection has bounding boxes
    - more data → simpler algs, less hand-engineering
    - less data → more hand-engineering, “hacks”
    - Two sources of knowledge
        - labeled data
        - hand engineered features/network architecture/other components
    - the state of CV has relied heavily on the later because of a lack of data
- Tips for doing well on benchmarks/winning competitions
    - usually using tricks that aren’t for production
    - ensembling
        - train several networks independently and average their outputs
        - 3 to 15 networks, but slows down by that much
    - multi-crop at test time
        - run classifier on multiple versions of test images and average results
        - 10-crop, 10 crops of one image

        - good for doing well on benchmarks
- not used for production systems
- Use open source code
    - use architectures of networks published int he literature
    - use open source implementation if possible
    - use pretrained models and fine-tune on personal dataset