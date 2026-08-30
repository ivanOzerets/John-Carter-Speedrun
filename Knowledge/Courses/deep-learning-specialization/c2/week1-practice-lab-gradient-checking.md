Packages

```python
import numpy as np
from testCases import *
from public_tests import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

%load_ext autoreload
%autoreload 2
```

Problem Statement

- part of a team working to make mobile payments available globaly
- asked to build a deep learning model to detect fraud - whenever someone makes a payment, you want to see if they payment might be fraudulent, such as if the user’s account has been taken over by a hacker
- you already know that backprop is quite challenging to implement, and sometimes has bugs
- make certain that the implementation of backprop is correct

How does Gradient Checking work?

- backprop compute the gradients of J w.r.t theta, where theta denotes the parameters of the model
- J is computed using forward prop and the loss function
- because forward prop is relatively easy, J is correcy in this case
- definition of a derivative (gradient)

- derivative of theta w.r.t theta is what is being checked
- compute J(theta + epsilon) and J(theta + epsilon)

1-Dimensional Gradient Checking

- consider a 1D linear function J(theta) = thetax
- the model contains only a single real-valued parameter theta, and takes x as input
- implement code to compute J(.) and its derivative w.r.t theta
- use gradient checking to make suer the derivative computation for J is correct

- the diagram shows the key computation steps
    - first start with x
    - evaluate the function J(x)
    - compute the derivative w.r.t theta
    - Exercise 1 - forward_prop
        - implement forward prop

        ```python
        # GRADED FUNCTION: forward_propagation

        def forward_propagation(x, theta):
            """
            Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)

            Arguments:
            x -- a real-valued input
            theta -- our parameter, a real number as well

            Returns:
            J -- the value of function J, computed using the formula J(theta) = theta * x
            """

            # (approx. 1 line)
            # J = 
            # YOUR CODE STARTS HERE
            J = theta * x
            # YOUR CODE ENDS HERE

            return J
        ```

    - Exercise 2 - backward_prop
        - implement the backward prop step
        - compute the derivative of J(theta) = thetax w.r.t theta
        - should get dtheta = dJ/dtheta = x

        ```python
        # GRADED FUNCTION: backward_propagation

        def backward_propagation(x, theta):
            """
            Computes the derivative of J with respect to theta (see Figure 1).

            Arguments:
            x -- a real-valued input
            theta -- our parameter, a real number as well

            Returns:
            dtheta -- the gradient of the cost with respect to theta
            """

            # (approx. 1 line)
            # dtheta = 
            # YOUR CODE STARTS HERE
            dtheta = x    
            # YOUR CODE ENDS HERE

            return dtheta
        ```

    - Exercise 3 - gradient_chck
        - show that the backward_prop() function is correctly computing the gradient
        - first compute gradapprox using the formal above and a small value of epsilon

        - compute the gradient using backward prop, and store the result in a variable “grad”
        - compute the relative difference between “gradapprox” and the “grad”

        - to compute
            - compute the numerator using np.linalg.norm()
            - compute the denominator, need to call np.linalg.norm() twice
            - divide them
        - if the difference is small (less that q0^-7), confident that gradient is computed correctly

        ```python
        # GRADED FUNCTION: gradient_check

        def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
            """
            Implement the gradient checking presented in Figure 1.

            Arguments:
            x -- a float input
            theta -- our parameter, a float as well
            epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

            Returns:
            difference -- difference (2) between the approximated gradient and the backward propagation gradient. Float output
            """

            # Compute gradapprox using right side of formula (1). epsilon is small enough, you don't need to worry about the limit.
            # (approx. 5 lines)
            # theta_plus =                                 # Step 1
            # theta_minus =                                # Step 2
            # J_plus =                                    # Step 3
            # J_minus =                                   # Step 4
            # gradapprox =                                # Step 5
            # YOUR CODE STARTS HERE
            theta_plus = theta + epsilon
            theta_minus = theta - epsilon
            J_plus = forward_propagation(x, theta_plus)
            J_minus = forward_propagation(x, theta_minus)
            gradapprox = (J_plus - J_minus) / (2 * epsilon)
            # YOUR CODE ENDS HERE

            # Check if gradapprox is close enough to the output of backward_propagation()
            #(approx. 1 line) DO NOT USE "grad = gradapprox"
            # grad =
            # YOUR CODE STARTS HERE
            grad = backward_propagation(x, theta)    
            # YOUR CODE ENDS HERE

            #(approx. 3 lines)
            # numerator =                                 # Step 1'
            # denominator =                               # Step 2'
            # difference =                                # Step 3'
            # YOUR CODE STARTS HERE
            numerator = np.linalg.norm(grad - gradapprox)
            denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
            difference = numerator / denominator
            # YOUR CODE ENDS HERE
            if print_msg:
                if difference > 2e-7:
                    print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
                else:
                    print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

            return difference

         # difference = 7.814075313343006e-11
        ```

    - the difference is smaller than the threshold → high confidence
    - the cost function J has more than a single 1D input
    - when training a NN, theta actually consists of multiple matrices W[l] and biases b[l]

N-Dimensional Gradient Checking

- the following figure describes the forward and backward propagation of your fraud detection model

- look at implementations of forward prop and back prop

    ```python
    def forward_propagation_n(X, Y, parameters):
        """
        Implements the forward propagation (and computes the cost) presented in Figure 3.

        Arguments:
        X -- training set for m examples
        Y -- labels for m examples 
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (5, 4)
                        b1 -- bias vector of shape (5, 1)
                        W2 -- weight matrix of shape (3, 5)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)

        Returns:
        cost -- the cost function (logistic cost for m examples)
        cache -- a tuple with the intermediate values (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

        """

        # retrieve parameters
        m = X.shape[1]
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]

        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = relu(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = sigmoid(Z3)

        # Cost
        log_probs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
        cost = 1. / m * np.sum(log_probs)

        cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

        return cost, cache
    ```

    ```python
    def backward_propagation_n(X, Y, cache):
        """
        Implement the backward propagation presented in figure 2.

        Arguments:
        X -- input datapoint, of shape (input size, 1)
        Y -- true "label"
        cache -- cache output from forward_propagation_n()

        Returns:
        gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
        """

        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

        dZ3 = A3 - Y
        dW3 = 1. / m * np.dot(dZ3, A2.T)
        db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1. / m * np.dot(dZ2, A1.T)
        db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1. / m * np.dot(dZ1, X.T)
        db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                     "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                     "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

        return gradients
    ```

- compare “gradapprox” to the gradient computed by backprop

- however, theta is not a scalar anymore
- it is a dictionary called “parameters”
- the function “dictionary_to_vector()” converts the parameters dictionary into a vector called “values”, obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them
- the inverse function is vector_to_dictionary which outputs back the parameters dictionary

- the gradients dictionary has also ben converted into vector grad using gradients_to_vector()
- for every single parameter in the vector, apply the same procedure as for the gradient_check exercise
- store each gradient approx in a vector gradapprox
- if the check goes as expected, each value in this approx must match the real gradient values stored in the grad vector
- grad is calculated using the function gradients_to_vector, which uses the gradients outputs of the backward_propagation_n funciton
- Exercise 4 - gradient_check_n
    - implement the function
    - for each i in num_parameters
        - compute J_plus[i]
            - set theta_plus to np.copy(parameters_values)
            - set theta_plus to theta_plus + epsilon
            - calculate J_plus using forward_propagation_n(x, y, vector_to_dictionary(theta_plus))
        - compute J_minus[i]: do the same thing with theta_minus
        - compute gradapprox[i] = J_plus - J_minus / 2 epsilon
    - get a vector gradapprox, where gradapprox[i] is an approximation of the gradient w.r.t parameter_values[i]
    - now compare the gradapprox vector to the gradients vector from backprop

    ```python
    # GRADED FUNCTION: gradient_check_n

    def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
        """
        Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

        Arguments:
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
        grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters 
        X -- input datapoint, of shape (input size, number of examples)
        Y -- true "label"
        epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
        """

        # Set-up variables
        parameters_values, _ = dictionary_to_vector(parameters)

        grad = gradients_to_vector(gradients)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        # Compute gradapprox
        for i in range(num_parameters):

            # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
            # "_" is used because the function you have outputs two parameters but we only care about the first one
            #(approx. 3 lines)
            # theta_plus =                                        # Step 1
            # theta_plus[i] =                                     # Step 2
            # J_plus[i], _ =                                     # Step 3
            # YOUR CODE STARTS HERE
            theta_plus = np.copy(parameters_values)
            theta_plus[i] += epsilon
            J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))
            # YOUR CODE ENDS HERE

            # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
            #(approx. 3 lines)
            # theta_minus =                                    # Step 1
            # theta_minus[i] =                                 # Step 2        
            # J_minus[i], _ =                                 # Step 3
            # YOUR CODE STARTS HERE
            theta_minus = np.copy(parameters_values)
            theta_minus[i] -= epsilon
            J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))        
            # YOUR CODE ENDS HERE

            # Compute gradapprox[i]
            # (approx. 1 line)
            # gradapprox[i] = 
            # YOUR CODE STARTS HERE
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)        
            # YOUR CODE ENDS HERE

        # Compare gradapprox to backward propagation gradients by computing difference.
        # (approx. 3 line)
        # numerator =                                             # Step 1'
        # denominator =                                           # Step 2'
        # difference =                                            # Step 3'
        # YOUR CODE STARTS HERE
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator
        # YOUR CODE ENDS HERE
        if print_msg:
            if difference > 2e-7:
                print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
            else:
                print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

        return difference

    # There is a mistake in the backward propagation! difference = 0.2850931567761623
    ```

- is seems that there were errors in the backward_propagation_n code
- find error in dW2 and db1
- gradient checking is slow
- only run a few times
- doesn’t work with dropout