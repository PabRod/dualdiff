from dualdiff.dual import Dual
from dualdiff.primitives import *
from dualdiff.decorators import autodifferentiable
import numpy as np
from pytest import approx


def test_sin():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return sin(x) + sin(1)

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return np.cos(x)

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == df_ref(c)


def test_cos():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return cos(x) - cos(np.pi)

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return -sin(x)

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == df_ref(c)


def test_composition():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return cos(sin(x**2))

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return -sin(sin(x**2)) * cos(x**2) * 2 * x

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == approx(df_ref(c))

def test_tan():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return tan(x)

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return 1/cos(x)**2

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == approx(df_ref(c))


def test_exp():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return exp(3*x)

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return 3*exp(3*x)

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == approx(df_ref(c))


def test_funpow():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return exp(x)**cos(x)

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return exp(x * cos(x)) * (cos(x) - x * sin(x))

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == approx(df_ref(c))


def test_verbose():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        v = 0
        for n in [0, 1, 2, 3]:
            v = v + x ** n

        return v

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return 3 * x**2 + 2 * x + 1

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == approx(df_ref(c))


def test_piecewise():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        if x < 0:
            return x ** 2
        else:
            return x ** 3

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): 
        if x < 0:
            return 2 * x
        else:
            return 3 * x ** 2

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == approx(df_ref(c))
