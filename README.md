# Newton

A package to solve non-linear equations

## Exemple

<img src="http://www.sciweavers.org/tex2img.php?eq=f%28x%29%20%3D%20%20x%5E%7B2%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="f(x) =  x^{2}" width="108" height="31" />

<p><p\>

<img src="http://www.sciweavers.org/tex2img.php?eq=J%20%5Cbig%5C%7Bf%28x%29%20%5Cbig%5C%7D%20%3D%20%20f%27%20%3D%202x&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0" align="center" border="0" alt="J \big\{f(x) \big\} =  f' = 2x" width="212" height="33" />


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
print(converged, error, solution)
```

## Documentation

``` python
import newton
help(newton)
```
