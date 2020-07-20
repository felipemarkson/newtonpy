import newtonpy
import inspect
import numpy as np
from numpy import ndarray
from typing import Tuple, Callable, Union


def test_newtonpy_has_solve():
    assert hasattr(newtonpy, "solve")


def test_newtonpy_solve_is_callabe():
    inspect.isfunction(newtonpy.solve)


class TestSolveSignature:
    sig = inspect.signature(newtonpy.solve)

    def test_solve_return_tuple(cls):
        assert cls.sig.return_annotation is Tuple[bool, ndarray, ndarray]

    def test_solve_has_param_func(cls):
        assert cls.sig.parameters["func"]

    def test_solve_param_func_is_callable(cls):
        assert cls.sig.parameters["func"].annotation is Callable[[ndarray], ndarray]

    def test_solve_has_param_jacobian(cls):
        assert cls.sig.parameters["jacobian"]

    def test_solve_param_jacobian_is_callable(cls):
        assert cls.sig.parameters["jacobian"].annotation is Callable[[ndarray], ndarray]

    def test_solve_has_param_x0(cls):
        assert cls.sig.parameters["x0"]

    def test_solve_param_x0_is_ndarray(cls):
        assert cls.sig.parameters["x0"].annotation is np.ndarray

    def test_solve_has_param_tol(cls):
        assert cls.sig.parameters["tol"]

    def test_solve_param_tol_is_union(cls):
        assert cls.sig.parameters["tol"].annotation is Union[int, float, ndarray]

    def test_solve_has_param_maxiter(cls):
        assert cls.sig.parameters["maxiter"]

    def test_solve_param_maxiter_is_int(cls):
        assert cls.sig.parameters["maxiter"].annotation is int

    def test_solve_has_param_verbose(cls):
        assert cls.sig.parameters["verbose"]

    def test_solve_param_verbose_is_bool(cls):
        assert cls.sig.parameters["verbose"].annotation is bool


class TestSolveOneVariable:
    def test_solve_converge(self):
        result = newtonpy.solve(
            lambda x: np.array([x ** 2]),
            lambda x: np.array([2 * x]),
            x0=np.array([1.2]),
            tol=0.001,
            maxiter=100,
            verbose=True,
        )

        assert result[0] is True
        assert result[1] <= 0.001 or result[1] >= -0.001
        assert result[2] <= 0.001 or result[2] >= -0.001

    def test_solver_not_coverge(self):
        result = newtonpy.solve(
            lambda x: np.array([x ** 2]),
            lambda x: np.array([2 * x]),
            x0=np.array([100]),
            tol=1e-10,
            maxiter=2,
            verbose=True,
        )

        assert result[0] is False


class TestSolveMultVariable:
    def test_solve_converge(self):
        result = newtonpy.solve(
            lambda x: np.array([x[0] ** 2 + x[1] ** 2, 2 * x[1]]),
            lambda x: np.array([[2 * x[0], 2 * x[1]], [0, 2]]),
            x0=np.array([1, 1]),
            tol=np.array([0.001, 0.001]),
            maxiter=100,
            verbose=True,
        )

        assert result[0] is True
        assert all(result[1] <= 0.001) or all(result[1] >= -0.001)
        assert all(result[2] <= 0.001) or all(result[2] >= -0.001)

    def test_solve_not_converge(self):
        result = newtonpy.solve(
            lambda x: np.array([x[0] ** 2 + x[1] ** 2, 2 * x[1]]),
            lambda x: np.array([[2 * x[0], 2 * x[1]], [0, 2]]),
            x0=np.array([100, 100]),
            tol=1e-10,
            maxiter=2,
            verbose=True,
        )

        assert result[0] is False
