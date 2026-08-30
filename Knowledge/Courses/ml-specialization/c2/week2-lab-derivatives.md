Informal definition of derivatives

- the formal definition of derivates can be a bit daunting with limits and values ‘gong to zero’
- the idea is really much simpler
- the derivative of a function describes how the output of a function changes when there is a small change in an input variable
- use the cost function J(w) as an example
- the cost J is the output and w is the input variable
- give a ‘small change’ a name epsilon
- traditional in mathematics to use epsilon or delta to represent a small value
- think of it as representing 0.001 or some other small value

- if you change the input to the function J(w) by a little bit and the output changes by k times that little bit, then the derivative of J(w) is equal to k
- look at the derivative of the function J(w) = w^2 at the point w = 3 and epsilon = 0.001

    ```python
    J = (3)**2
    J_epsilon = (3 + 0.001)**2
    k = (J_epsilon - J)/0.001    # difference divided by epsilon
    print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")

    # J = 9, J_epsilon = 9.006001, dJ_dw ~= k = 6.001000 
    ```

- increasing the input value a little bit (0.001), causes the output to change from 9 to 9.006001, an increase of six times the input increase
- so k = 6
- so derivative of J w.r.t w ~ 6
- symbolically written through calculus as 2w
- with w = 3, this becomes 6
- the calculation above is not exactly 6 because epsilon was not infinitesimally small

    ```python
    J = (3)**2
    J_epsilon = (3 + 0.000000001)**2
    k = (J_epsilon - J)/0.000000001
    print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

    # J = 9, J_epsilon = 9.000000006, dJ_dw ~= k = 6.000000496442226 
    ```

Finding symbolic derivatives

- in backprop it is useful to know the derivative of simple functions at any input value
- put another way, we would like to know the ‘symbolic’ derivative rather than the ‘arithmetic’ derivative
- an example of a symbolic derivative is the 2w, the derivative of J(w) = w^2
- with the symbolic derivative you can find the value of the derivative at any input value w
- this process can be automated with symbolic differentiation programs
- an example is the with the SymPy library
- J = w^2
- define the python variables and their symbolic names

    ```python
    J, w = symbols('J, w')
    ```

- define and print the expression
- note sympy produces a latex string which generates a nicely readable equation

    ```python
    J=w**2
    J

    # 𝑤^2
    ```

- use SymPy’s diff to differentiate the expression for w.r.t w
- note the result matches the earlier example

    ```python
    dJ_dw = diff(J,w)
    dJ_dw

    # 2w
    ```

- Evaluate the derivative at a few points by ‘substituting’ numeric values for the symbolic values
- in the first example, w is replaced by 2

    ```python
    dJ_dw.subs([(w,2)])    # derivative at the point w = 2

    # 4
    ```

    ```python
    dJ_dw.subs([(w,3)])    # derivative at the point w = 3

    # 6
    ```

    ```python
    dJ_dw.subs([(w,-3)])    # derivative at the point w = -3

    # -6
    ```

- with J = 2w

    ```python
    w, J = symbols('w, J')
    J = 2 * w
    J

    # 2w
    ```

    ```python
    dJ_dw = diff(J,w)
    dJ_dw

    # 2
    ```

    ```python
    dJ_dw.subs([(w,-3)])    # derivative at the point w = -3

    # 2
    ```

- compared with the arithmetic calculation

    ```python
    J = 2*3
    J_epsilon = 2*(3 + 0.001)
    k = (J_epsilon - J)/0.001
    print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

    # J = 6, J_epsilon = 6.002, dJ_dw ~= k = 1.9999999999997797
    ```

- for the function J = 2w, it is easy to see that any change in w will result in 2 times that amount of change in the output J, regardless of the starting value of w
- for j = w^3

    ```python
    J, w = symbols('J, w')
    J=w**3
    J

    # w^3
    ```

    ```python
    dJ_dw = diff(J,w)
    dJ_dw

    # 3w^2
    ```

    ```python
    dJ_dw.subs([(w,2)])   # derivative at the point w=2

    # 12
    ```

- compared to the arithmetic calculation

    ```python
    J = (2)**3
    J_epsilon = (2+0.001)**3
    k = (J_epsilon - J)/0.001
    print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

    # J = 8, J_epsilon = 8.012006000999998, dJ_dw ~= k = 12.006000999997823 
    ```

- if J = 1/w

    ```python
    J, w = symbols('J, w')
    J= 1/w
    J

    # 1/w
    ```

    ```python
    dJ_dw = diff(J,w)
    dJ_dw

    # -1/w^2
    ```

    ```python
    dJ_dw.subs([(w,2)])

    # -1/4
    ```

- compared to the arithmetic calculation

    ```python
    J = 1/2
    J_epsilon = 1/(2+0.001)
    k = (J_epsilon - J)/0.001
    print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

    # J = 0.5, J_epsilon = 0.49975012493753124, dJ_dw ~= k = -0.2498750624687629 
    ```

- if J = 1/w^2

    ```python
    J, w = symbols('J, w')
    J = 1/w**2
    J

    # 1/w^2
    ```

    ```python
    dJ_dw = diff(J,w)
    dJ_dw

    # -2/w^3
    ```

    ```python
    dJ_dw.subs([(w,2)])

    # -1/4
    ```

- compared to the arithmetic calculation

    ```python
    J = 1/4**2
    J_epsilon = 1/(4+0.001)**2
    k = (J_epsilon - J)/0.001
    print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

    # J = 0.0625, J_epsilon = 0.06246876171484496, dJ_dw ~= k = -0.031238285155041345 
    ```