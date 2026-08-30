Computer Vision

- Computer Vision Problems
    - image classification
    - object detection
    - Neural Style Transfer → repaint content image in a certain style
- Deep learning on large images
    - challenge → images can get very big
    - for images of size 1000x1000, the weight matrix will have 1000, 3 million = 3 billion parameters

Edge Detection Example

- Computer Vision Problem
    - detection horizontal and vertical edges

- Vertical edge detection
    - filter or kernel
    - convolution operation is the * notion, overloaded
    - to compute the first element, paste the filter over the original image
    - compute the element wise product and then add
    - shift the filter over and repeat

    - this turns out to be a vertical edge detector
    - python: conv-forward
    - tensorflow: tf.nn.conv2d
    - keras: conv2D
    - the more similar the values, the closer to zero
    - the more different across the vertical, the higher the value

More Edge Detection

- difference between positive and negative edges
- learn an edge detector
- Vertical edge detection examples
    - brighter mean higher number

- Vertical and Horizontal Edge Detection
    - horizontal edge would be across on the filter

- Learning to detect edges
    - what are the best numbers to use
    - sobel filter → more weight to the middle
    - scharr filter also another filter

    - treat the numbers of the filter as parameters
    - could learn to detect at different angles

Padding

- modification to basic convolution
- 6x6 convolved with 3x3 ends up with 4x4
- only 4x4 possible positions for the filter to fit in the 6x6 matrix
- nxn * fxf = n-f+1 x n-f+1
- downsides
    - every time filter is applied, the image shrinks
    - a corner pixel is used a lot less than a center pixel
- Padding
    - pad the image with an additional pixel
    - you preserve the initial image size
    - can pad with many pixels
- Valid and Same convolutions
    - valid
        - no padding
        - nxn * fxf → n-f+1 x n-f+1
    - same
        - pad so that output size is the same as the input size
        - n+2p-f+1 x n+2p-f+1
        - p = (f-1) / 2 for same size
        - f is usually odd
        - even f needs an asymmetric pattern

Strided Convolutions

- skip over by multiple steps
- stride length
- nxn * fxf with padding p and stride s

- round down as to not look at extra non-existent pixels
- Technical not on cross-correlation vs convolution
    - convolution in math textbook
    - first flip horizontally and vertically
    - in ML, skip the step → cross-correlation
    - in branches of mathematics, doing the flipping give the following property

Convolutions Over Volume

- Convolutions on RGB images
    - stack of images
    - convolve with stacked filter
    - height, width, and number of channels
    - filer and images channels must match
    - output is a single image, one channel

    - edges in red channel, blue and green channel be all zeros
    - filter that detects edges in the red channel
- Multiple filters
    - two filter outputs stacked

    - nxnxnc * fxfxnc → n-f+1 x n-f+1 x nc’
    - channels known as depth as well

One Layer of a Convolutional Network

- Example of a layer
    - add bias to the output from the convolution
    - then apply non-linearity
    - outputs same shape

    - Z[1] = W[1]a[0] + b[1]
    - a[1] = g(z[1])
    - filter plays the role of W[1]
    - the output of the convolution, plays the role of W[1]a[0]
    - Z[1] is adding the bias
    - a[1] is stacking the Zs
- Number of parameters in one layer
    - if you have 10 filters that are 3x3x3 in one layer of a NN, how many parameters does that layer have? 270 + 10 for bias = 280
    - number of parameters is fixed in correlation to the size of the image
- Summary of notation
    - layer l is a convolution layer
    - f[l] = filter size
    - p[l] = padding
    - s[l] = stride
    - input: nh[l-1] x nw[l-1] x nc[l-1]
    - output: nh[l] x nw[l] x nc[l]

    - nc[l] = number of filters
    - each filter: f[l] x f[l] x nc[l-1]
    - activations: a[l] → nh[l] x nw[l] x nc[l]
    - with batch: A[l] → m x nh[l] x nw[l] x nc[l]
    - weights: f[l] x f[l] x nc[l-1] x nc[l]
    - bias: nc[l] → (1,1,1,n[l])
    - not standard notation, channels could be first or last

Simple Convolutional Network Example

- going deeper
- Example ConvNet
    - flatten last layer into one vector
    - feed into softmax layer

    - a lot of the work is selecting the parameters
    - typically the height and width might stay the same for a while, and then gradually decrease
- Types of layer in a convolutional network
    - convolution (conv)
    - pooling (pool)
    - fully connected (fc)

Pooling layers

- Max pooling
    - take the input, break into region, then take the max of the regions
    - hypers of max pooling
        - f = 2
        - s = 2
    - a large number might mean that the model has detected a particular feature
    - so the upper left-hand quadrant has this particular feature
    - it remains preserved through max pooling
    - no parameters to learn
- Example
    - filter 3
    - stride 1
    - same formulas for output size for conv layers work for pooling layers

    - output of pooling will have same dimension of the input
    - results in the same number of channels
- Average pooling
    - not used very often
    - take the average of the regions
    - can use very deep to average over all to get 1x1x..
- hypers
    - filter size
    - stride
    - max or average pooling
    - p = 0, rarely used
    - f = 2, s = 2
    - f = 3, s = 2 is common

CNN Example

- NN example
- close to LeNet-5 by parameter choice
- conventions
    - layer 1 is conv and pool layer together
    - conv and pool layers are separate layers
    - since pool layer has no weights, layer 1 is conv and pool layer together
- not writing padding, means no padding

- common pattern is conv → pool → conv → pool → fc → fc → fc → softmax
- conv have relatively layers
- most parameters are in the fully connected layers

Why Convolutions?

- advantages
    - parameter sharing
    - sparsity of connections
- weight matrix parameter count for a fully connected network is way greater than for conv layers
- parameter sharing: a feature detector (such as a vertical edge detector) that’s useful in one part of the image is probably useful in another part of the image
- intuitive: same feature useful for many parts of the image
- sparsity of connections: in each layer, each output value depends only on a small number of inputs
- less prone to overfitting
- translation invariance
- Putting it together
    - training set (x1,y1) … (xm, ym)
    - use GD to optimize parameters to reduce J

Yann LeCun Interview

- LeNet creator
- CRF → interpreting sequence of characters
- contribute to open source contributions