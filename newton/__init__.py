import numpy as np
from typing import Union, Tuple, Callable

Vec = Union[int, float, np.ndarray]

Func = Callable[[Vec], Vec]

Result = Tuple[bool, Vec, Vec]


def solve(
    func: Func, jacobian: Func, *, x0: Vec, tol: Vec, maxiter: int, verbose: bool
) -> Result:
    pass
