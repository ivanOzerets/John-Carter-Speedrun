- GD requires the derivative of the cost with respect to each parameter in the network
- NN can have millions or even billions of parameters
- the back propagation alg is used to compute those derivatives
- computation graphs are used to simplify the operation

    ```python
    from sympy import *
    import numpy as np
    import re
    %matplotlib widget
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox
    from matplotlib.widgets import Button
    import ipywidgets as widgets
    from lab_utils_backprop import *
    ```

Computation Graph

- a computation graph simplifies the computation of complex derivatives by breaking them into smaller steps
- calculate the derivative of this slightly complex expression, J = (2 + 3w)^2
- find the derivative of J w.r.t w

- we broke the expression into two nodes which we can work on independently
- first fill in the blue boxes going left to right and then fill in the green boxes starting on the right and moving to the left

Forward Propagation

- calculate the values in the forward directions
    - this section uses global variables and resuses them as the calculation progresses

    ```python
    w = 3
    a = 2+3*w
    J = a**2
    print(f"a = {a}, J = {J}")

    # a = 11, J = 121
    ```

- same values as above

Backprop

- backprop is the alg to calculate derivatives
- backprop starts at the right and moves to the left
- the first node to consider is J = a^2 and the first step is to find the derivative of J w.r.t a
- Arithmetically
    - find the derivative of J w.r.t. a by finding how J changes as a result of a little change in a

        ```python
        a_epsilon = a + 0.001       # a epsilon
        J_epsilon = a_epsilon**2    # J_epsilon
        k = (J_epsilon - J)/0.001   # difference divided by epsilon
        print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

        # J = 121, J_epsilon = 121.02200099999999, dJ_da ~= k = 22.000999999988835 
        ```

    - the derivative is 22 which is 2 x a
    - the result is not exactly 2 x a because the epsilon value is not infinitesimally small
- Symbolically
    - use sympy to calculate derivatives symbolically
    - prefitx the name of the variable with an ‘s’ to indicate a symbolic variable

        ```python
        sw,sJ,sa = symbols('w,J,a')
        sJ = sa**2
        sJ

        # a^2
        ```

        ```python
        sJ.subs([(sa,a)])

        # 121
        ```

        ```python
        dJ_da = diff(sJ, sa)
        dJ_da

        # 2a
        ```

    - the derivative is 2a
    - when a = 11, the derivative is 22
    - this matches the arithmetic calculation above
    - moving from right to left, the next value is the derivative of J w.r.t w
    - first calculate the derivative of a w.r.t. a which describes how the output of the node, a, changes when the input w changes a little bit
- Arithmetically
    - find the derivative a w.r.t w by finding how a changes as a result of a little change in w

        ```python
        w_epsilon = w + 0.001       # a  plus a small value, epsilon
        a_epsilon = 2 + 3*w_epsilon
        k = (a_epsilon - a)/0.001   # difference divided by epsilon
        print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")

        # a = 11, a_epsilon = 11.003, da_dw ~= k = 3.0000000000001137 
        ```

    - calculated arithmetically, the derivative is ~3
- Symbolically
    - again using sympy

        ```python
        sa = 2 + 3*sw
        sa

        # 3w + 2
        ```

        ```python
        da_dw = diff(sa,sw)
        da_dw

        # 3
        ```

- the next step is the interesting part:
    - we know that a small change in w will cause a to change by 3 times that amount
    - we know that a small change in a will cause J to change by 2 x a times that amount
    - we know that a small change in w will cause J to change by 3 x 2 x a times that amount
- these cascading changes go by the name of the chain rule

    ```python
    dJ_dw = da_dw * dJ_da
    dJ_dw

    # 6a
    ```

- a is 11, so the derivative of J w.r.t. w is 66
- checking arithmetically:

    ```python
    w_epsilon = w + 0.001
    a_epsilon = 2 + 3*w_epsilon
    J_epsilon = a_epsilon**2
    k = (J_epsilon - J)/0.001   # difference divided by epsilon
    print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

    # J = 121, J_epsilon = 121.06600900000001, dJ_dw ~= k = 66.0090000000082 
    ```

- Another view
    - one could visualize the cascading changes this way:

    - a small change in w is multiplied by the derivative of a w.r.t. w, resulting in a change that is 3 times as large
    - this larger change is then multiplied by the derivative of J w.r.t. a, resulting in a change that is now 3 x 22 = 66 times larger

Computation Graph of a Simple NN

- below is a graph of a NN
- try and fill in the values in the boxes

Forward propagation

- compare the values below to the one above

    ```python
    # Inputs and parameters
    x = 2
    w = -2
    b = 8
    y = 1
    # calculate per step values   
    c = w * x
    a = c + b
    d = a - y
    J = d**2/2
    print(f"J={J}, d={d}, a={a}, c={c}")

    # J=4.5, d=3, a=4, c=-4
    ```

Backward propagation (Backprop)

- Arithmetically
    - find the derivative of J w.r.t d by finding how J changes as a result of a little change in d

        ```python
        d_epsilon = d + 0.001
        J_epsilon = d_epsilon**2/2
        k = (J_epsilon - J)/0.001   # difference divided by epsilon
        print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dd ~= k = {k} ")

        # J = 4.5, J_epsilon = 4.5030005, dJ_dd ~= k = 3.0004999999997395 
        ```

    - the derivative is 3, which is the value of d
    - the result is not exact because of the epsilon value
- Symbolically
    - using sympy

        ```python
        sx,sw,sb,sy,sJ = symbols('x,w,b,y,J')
        sa, sc, sd = symbols('a,c,d')
        sJ = sd**2/2
        sJ

        # d^2/2
        ```

        ```python
        sJ.subs([(sd,d)])

        # 9/2
        ```

        ```python
        dJ_dd = diff(sJ, sd)a

        # d
        ```

    - the derivative of J w.r.t d = d
    - when d = 3 → the derivative = 3
- Arithmetically
    - find the derivative of d w.r.t. a by finding how d changes as a result of a little change in a

        ```python
        a_epsilon = a + 0.001         # a  plus a small value
        d_epsilon = a_epsilon - y
        k = (d_epsilon - d)/0.001   # difference divided by epsilon
        print(f"d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ")

        # d = 3, d_epsilon = 3.0010000000000003, dd_da ~= k = 1.000000000000334 
        ```

    - arithmetically, the derivative is ~ 1
- Symbolically
    - again with sympy

        ```python
        sd = sa - sy
        sd

        # a-y
        ```

        ```python
        dd_da = diff(sd,sa)
        dd_da

        # 1
        ```

- calculated arithmetically, the derivative also equals 1
    - the next step is the interesting part
        - we know that a small change in a will cause d to change by 1 times that amount
        - we know that a small change in d will cause J to change by d times that amount
        - we know that a small change in a will case J to change by 1 x d times that amount
    - this is again the chain rule
- calculating it:

    ```python
    dJ_da = dd_da * dJ_dd
    dJ_da

    # d
    ```

- and d is 3, so the derivative of  w.r.t. a is 3
- checking arithmetically

    ```python
    ae = a + 0.001
    de = ae - y
    Je = de**2/2
    k = (Je - J)/0.001   
    print(f"J = {J}, J epsilon = {Je}, dJ_da ~= k = {k} ")

    # J = 4.5, J epsilon = 4.503000500000001, dJ_da ~= k = 3.0005000000006277 
    ```

- the steps in backprop
    - calculate the local derivatives of the node
    - using the chain rule, combine with the derivative of the cost w.r.t the node to the right
- the local derivatives are the derivatives of the output of the current node w.r.t all inputs or parameters
- the next node has two derivatives of interest
- calculate the derivative of J w.r.t c to be able to propagate to the left
- also calculate the derivative of J w.r.t b
- finding the derivative of the cost w.r.t the parameters w and b is the object of backprop
- we will find the local derivatives, w.r.t c and b first and then combine those with the derivative coming from the right

    ```python
    # calculate the local derivatives da_dc, da_db
    sa = sc + sb
    sa

    # b + c
    ```

    ```python
    da_dc = diff(sa,sc)
    da_db = diff(sa,sb)
    print(da_dc, da_db)

    # 1 1
    ```

    ```python
    dJ_dc = da_dc * dJ_da
    dJ_db = da_db * dJ_da
    print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")

    # dJ_dc = d,  dJ_db = d
    ```

- in the example, d = 3
- the last node calculates c
- calculate how J changes w.r.t to parameter w
- not back propagating to the input x, so not interesting in the derivative of J w.r.t x

    ```python
    # calculate the local derivative
    sc = sw * sx
    sc

    # wx
    ```

    ```python
    dc_dw = diff(sc,sw)
    dc_dw

    # x
    ```

- this derivative depends on the value of x

    ```python
    dJ_dw = dc_dw * dJ_dc
    dJ_dw

    # dx
    ```

    ```python
    print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")

    # dJ_dw = 2*d
    ```

- d = 3, so the derivative of J w.r.t w = 6
- arithmetically:

    ```python
    Je = ((w+0.001)*x+b - y)**2/2
    k = (Je - J)/0.001  
    print(f"J = {J}, J epsilon = {Je}, dJ_dw ~= k = {k} ")

    # J = 4.5, J epsilon = 4.506002, dJ_dw ~= k = 6.001999999999619 
    ```