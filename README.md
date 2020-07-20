# Newton

A package to solve nonlinear equations by Newton–Raphson method

## Exemple
-----

### One variable

The function:

<img src="http://www.sciweavers.org/tex2img.php?eq=f%28x%29%20%3D%20%20x%5E%7B2%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="f(x) =  x^{2}" width="108" height="31" />


The Jacobian of function:

<img src="http://www.sciweavers.org/tex2img.php?eq=J%20%5Cbig%5C%7Bf%28x%29%5Cbig%5C%7D%20%3D%20%20%5Cbegin%7Bbmatrix%7D2x%20%5Cend%7Bbmatrix%7D%20%20&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="J \big\{f(x)\big\} =  \begin{bmatrix}2x \end{bmatrix}  " width="175" height="33" />


``` python
import newton
import numpy as np

(converged, error, solution) = newton.solve(
    lambda x: x ** 2,
    lambda x: np.array([2 * x]),
    x0=np.array([1.2]),
    tol=0.001,
    maxiter=100,
)
print(solution)
```

### Multivariable

The function:

<img src="http://www.sciweavers.org/tex2img.php?eq=f%28%20%5Coverrightarrow%7Bx%7D%29%20%3D%20%20%5Cbegin%7Bbmatrix%7D%20%20x_%7B0%7D%5E2%20%2B%20x_%7B1%7D%5E2%20%5C%5C%202x_%7B1%7D%20%5Cend%7Bbmatrix%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="f( \overrightarrow{x}) =  \begin{bmatrix}  x_{0}^2 + x_{1}^2 \\ 2x_{1} \end{bmatrix}" width="200" height="62" />

The Jacobian of function:

<img src="http://www.sciweavers.org/tex2img.php?eq=%20J%5Cbig%5C%7B%20f%28%20%5Coverrightarrow%7Bx%7D%29%20%5Cbig%5C%7D%20%3D%20%20%5Cbegin%7Bbmatrix%7D%20%202x_0%20%26%202x_1%20%5C%5C%200%20%26%202%20%5Cend%7Bbmatrix%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt=" J\big\{ f( \overrightarrow{x}) \big\} =  \begin{bmatrix}  2x_0 & 2x_1 \\ 0 & 2 \end{bmatrix}" width="258" height="62" />

``` python
import newton
import numpy as np

(converged, error, solution) = newton.solve(
    lambda x: np.array([x[0] ** 2 + x[1] ** 2, 2 * x[1]]),
    lambda x: np.array([[2 * x[0], 2 * x[1]], [0, 2]]),
    x0=np.array([1, 1]),
    tol=1e-3,
    maxiter=10,
    verbose=True,
)
print(solution)
```


## Documentation
-----

``` python
import newton
help(newton)
```


## License and Copyright
-----
 
MIT License

Copyright (c) 2020 Felipe M. S. Monteiro (<fmarkson@outlook.com>)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---






