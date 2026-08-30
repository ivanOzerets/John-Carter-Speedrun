ReLU Activation

- Rectified Linear Unit
- a = max(0,z)

- a derived “awareness” feature is not binary but has a continuous range of values
- the sigmoid is best for on/off or binary situations
- the ReLU provides a continuous linear relationship
- additionally, it has an ‘off’ range where the output is zero
- the “off” feature makes the ReLU a non-linear activation

Why Non-Linear Activations

- the function below is composed of linear pieces (piecewise linear)
- the slope is consistent during the linear portion and then changes abruptly at transition points
- at transition points, a new linear function is added which, when added to the existing function, will produce the new slope
- the new function is added at transition point but does not contribute to the output prior to that point
- the non-linear activation function is responsible for disabling the input prior to and sometimes after the transition points

- the exercise will use the network below in a regression problem where you must model a piecewise linear target:

- the network has 3 units in the first layer
- each is required to form the target
- unit 0 is preprogrammed and fixed to map the first segment
- you will modify weights and biases in unit 1 and 2 to model the 2nd and 3rd segment
- the output unit is also fixed and simply sums the outputs of the first layer
- using the sliders, modify weights and biases to match the target

- the goal of the exercise is to appreciate how the ReLU’s non-linear behavior provides the needed ability to turn functions off until they are needed
- the plots on the right contain the output of the units in the first layer
- starting at the top, unit 0 is responsible for the first segment
- both the linear function z and the function following the ReLU a are shown
    - you can see that the ReLU cuts off the function after the interval [0,1]
    - this is important as it prevents unit 0 from interfering with the following segment
- Unit 1 is responsible for the 2nd segment
    - here the ReLU kept this unit quiet until x is 1
    - since the first unit is not contributing, the slope for unit 1, w_z^[1], is just the slope of the target line
    - the bias must be adjusted to keep the output negative until x has reached 1
    - note how the contribution of unit 1 extends to the 3rd segment as well
- Unit 2 is responsible for the 3rd segment
    - the ReLU again zeros the output until x reaches the right value
    - the slope of the unit, w_2^[1], must be set so that the sum of the unit 1 and 2 have the desired slope
    - the bias is again adjusted to keep the output negative until x has reached 2
- the “off” or disable feature of the ReLU activation enables models to stitch together linear segments to model complex non-linear functions