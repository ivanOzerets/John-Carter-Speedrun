Mini-batch Gradient Descent

- deep learning works best with big data
- training NN with a lot of data is slow]
- Batch vs. mini-bath gradient descent
    - vectorization allows for efficient computation on m examples
    - if M is very large, vectorization can still be slow
    - so far, the whole training set needs to be processed before taking a step in GD
    - better if GD can start to make some progress
    - split up training set into mini batches
    - and split for y, corresponding input output pairs
    - i is per training example ()
    - l is per layer []
    - t is per mini-batch {}
    - batch refers to processing all the training data all at the same time
    - mini-bath refers to alg which processes a part of the training set at a time
- Mini-batch gradient descent
    - inside a for loop, perform GD on input output data over X{t}, Y{t}
    - l is also overloaded to represent the number of mini-batch in the sum
    - one epoch is a single pass through the training set
    - with just batch training, you take one GD step per epoch, rather that say 5000 with mini-batch

Understanding Min0batch Gradient Descent

- Training with mini batch gradient descent
    - it cost ever goes up, something is wrong with batch
    - with mini-batch, it might not go down with every iteration

- Choosing the min-batch size
    - if mini-batch size = m → batch gradient descent
    - if minibatch size = 1 → stochastic gradient descent, every example is its own mini-batch
    - stochastic will never converge
    - in practice: somewhere in between
    - batch → too long, huge processing every step
    - stochastic → lose speedup from vectorization

    - if small training set: use batch gradient decent (< 2000)
    - typical mini-batch sizes: 64, 128, 256, 512
    - make sure mini-batches fit in CPU/GPU memory
    - mini-batch size is another hyperparameter

Exponentially Weighted Averages

- Temperature in London
    - if you want to compute the moving average (trends) of the temperature
    - exponentially weighted average
    - v0 = 0
    - vt = betav(t-1) + (1-beta)theta_t
    - vt is approximately averaging over 1/1-beta day’s temperature
    - formula adapt more slowly when the temp changes because looking back further in time

Understanding exponentially weighted averages

- Exponentially weighted averages
    - influence of previous days gets less and less important
    - it takes about ten days for the sum to decay to 1/3
    - so you’re taking the current temperature as the most important, then the subsequent temps less until you get the weight around a third of that nth day or around 1 / epsilon
    - taking the daily temperature, multiplying it with an exponentially decaying function, then summing it up

- Implementing exponentially weighted averages
    - takes very little memory

    - why not just keep the last 10 days in memory?
    - takes too much memory
    - more complicated to implement
    - instead, just keep one line of the running average and update with each day

Bias Correction in Exponentially Weighted Averages

- Bias correction
    - when implementing the formula you get the purple curve rather than the green curve

    - the first few estimates are not very good estimates
    - the computed v starts of as a fraction of the current or last few days temps
    - modified to Vt/(1-beta^t)
    - as t becomes large, beta becomes closer to one, and Vt is more and more unaffected
    - only when in the beginning, the bias correction term of the denominator shifts the average back up

GD with Momentum

- almost always works faster than standard
- compute an exponentially weighted average of the gradients
- then use that gradient to update the weights instead
- Gradient descent example
    - oscillations slows down learning
    - what slower learning on the vertical, faster on the horizontal (for this example)
    - momentum:
        - on iteration t:
            - compute dW, db on current min-batch
            - Vdw = betaVdw + (1-beta)dW
            - Vdb = betaVdb + (1-beta)db
            - W = W - alpha Vdw, b:= b- alphaVdb

    - the derivative terms provide ‘acceleration to a ball’ rolling down hill and the vs are the ‘velocity’

- Implementation details
    - hyperparameters alpha and beta
    - beta is most commonly 0.9, about last 10 gradients

    - in practice, people don’t usually implement bias correction because after just 10 iterations, the average has ‘warmed up’
    - in literature, often the (1-beta) term gets omitted
    - VdW ends up being scaled by a factor of 1/(1-beta)
    - so the learning rate alpha needs to change by the factor
    - unintuitive because if you tune beta, you will need to retune alpha as a result

RMSprop

- root mean squared prop
- slow down learning in the axis that aren’t good, speeds up learning the axis that are good
- on iteration t:
    - compute dW, db on current mini-batch
    - SdW = betaSdW + (1-beta)dW^2 (element-wise)
    - same for db
    - W := w - alpha (dW/sqrt(SdW))
    - same for db

- your either dampening or promoting the update by the inverse of the root of the square of the update
- so if the update is large is one direction, it will get dampened
- if the update is small, it will get magnified

- dampening oscillations
- beta2 hyperparameter
- add small epsilon to ensure numerical stability and not divide by zero

- allows for a larger learning rate alpha

Adam Optimization Algorithm

- RMSprop and Adam has shown to work well on a wide range of deep learning architectures
- putting together momentum and RMSprop
- VdW = 0, SdW = 0, Vdb, Sdb = 0
- On iteration t:
    - compute dW, db using current mini-batch
    - you do add bias correction with Adam

- alpha needs to be tuned
- beta1: 0.9 (dW), first moment
- beta2: 0.999 (dW^2), second moment
- epsilon: 10^-8
- Adam: adaptive moment estimation

Learning Rate Decay

- slowly reduce learning rate over time
- mini-bath won’t really converge because alpha is a fixed number
- oscillation tends toward a smaller region

- set alpha based on epoch

- decay rate is another hyperparameter

- Other learning rate decay methods
    - exponential decay

    - manual decay, looks like learning rate slowed down, manually decrease learning rate
    - works on long training times
- lower on the list of things to change

The Problem of Local Optima

- Local optima in NN
    - what the idea was

    - in low dimensions you can easily spot local optima
    - in higher dimensions, most points are actually a saddle point with zero gradient

    - for a point to be a local optima, all 20,000 directions, for example, need to be going up and away, which is very small
    - more likely where the function bends down
- Problem of plateaus
    - can take a long time to get to a saddle point

    - unlikely to get stuck in bad local optima
    - plateaus can make learning slow

Yuanqing Lin Interview

- PCAs
- ImageNet
- AlexNet
- Pelo Pelo
- LDA