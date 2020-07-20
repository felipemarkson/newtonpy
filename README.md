# Newton

A package to solve nonlinear equations by Newton–Raphson method

## Exemple

### One variable

The function:

### One variable


The function:

![Equation 1](docs/eq1.png)


The Jacobian of function:

![Equation 2](docs/eq2.png)

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

![Equation 3](docs/eq3.png)

The Jacobian of function:

![Equation 4](docs/eq4.png)


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

``` python
import newton
help(newton)
```


## License and Copyright
 
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






