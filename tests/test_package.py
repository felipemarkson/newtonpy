import newton
import inspect
import numpy as np
from typing import Union, Tuple, Callable


def test_newton_has_solve():
    assert hasattr(newton, "solve")


def test_newton_solve_is_callabe():
    inspect.isfunction(newton.solve)


def test_newton_has_Vec():
    assert hasattr(newton, "Vec")


def test_newton_Vec_Type():
    assert newton.Vec is Union[int, float, np.ndarray]


def test_newton_has_Func():
    assert hasattr(newton, "Func")


def test_newton_Func_Type():
    assert newton.Func is Callable[[newton.Vec], newton.Vec]


def test_newton_has_Result():
    assert hasattr(newton, "Result")


def test_newton_Result_Type():
    assert newton.Result is Tuple[bool, newton.Vec, newton.Vec]


class TestSolveSignature:
    sig = inspect.signature(newton.solve)

    def test_solve_return_tuple(cls):
        assert cls.sig.return_annotation is newton.Result

    def test_solve_has_param_func(cls):
        assert cls.sig.parameters["func"]

    def test_solve_param_func_is_callable(cls):
        assert cls.sig.parameters["func"].annotation is newton.Func

    def test_solve_has_param_jacobian(cls):
        assert cls.sig.parameters["jacobian"]

    def test_solve_param_jacobian_is_callable(cls):
        assert cls.sig.parameters["jacobian"].annotation is newton.Func

    def test_solve_has_param_x0(cls):
        assert cls.sig.parameters["x0"]

    def test_solve_param_x0_is_Vec(cls):
        assert cls.sig.parameters["x0"].annotation is newton.Vec

    def test_solve_has_param_tol(cls):
        assert cls.sig.parameters["tol"]

    def test_solve_param_tol_is_union_Vec(cls):
        assert cls.sig.parameters["tol"].annotation is newton.Vec

    def test_solve_has_param_maxiter(cls):
        assert cls.sig.parameters["maxiter"]

    def test_solve_param_maxiter_is_int(cls):
        assert cls.sig.parameters["maxiter"].annotation is int

    def test_solve_has_param_verbose(cls):
        assert cls.sig.parameters["verbose"]

    def test_solve_param_verbose_is_bool(cls):
        assert cls.sig.parameters["verbose"].annotation is bool


class TestSolveOneVariable:
    def test_solve_converge():
        result = newton.solve(
            lambda x: x ** 2,
            lambda x: 2 * x,
            x0=1.2,
            tol=0.001,
            maxiter=100,
            verbose=True,
        )

        assert result[0] is True
        assert result[1] <= 0.001 or result[1] >= -0.001
        assert result[2] <= 0.001 or result[2] >= -0.001

    def test_solver_not_coverge():
        result = newton.solve(
            lambda x: x ** 2,
            lambda x: 2 * x,
            x0=1000,
            tol=1e-10,
            maxiter=2,
            verbose=True,
        )

        assert result[0] is False


class TestSolveMultVariable:
    def test_solve_converge():
        result = newton.solve(
            lambda x: np.array([[x[0, 0] ** 2 + x[1, 0] ** 2]]),
            lambda x: np.array([[x[0, 0] * 2, x[1, 0] * 2]]),
            x0=np.array([[0.5], [0.5]]),
            tol=0.001,
            maxiter=100,
            verbose=True,
        )

        assert result[0] is True
        assert all(result[1] <= 0.001) or all(result[1] >= -0.001)
        assert all(result[2] <= 0.001) or all(result[2] >= -0.001)

    def test_solve_not_converge():
        result = newton.solve(
            lambda x: x[0, 0] ** 2 + x[1, 0] ** 2,
            lambda x: np.array([[x[0, 0] * 2, 0], [0, x[1, 0] * 2]]),
            x0=np.array([[500], [500]]),
            tol=1e-10,
            maxiter=2,
            verbose=True,
        )

        assert result[0] is False
