from dualdiff.dual import Dual
from dualdiff.decorators import autodifferentiable
from numpy import linspace
from pytest import approx

def test_sq():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return x ** 2 + 1
    
    # Automatic derivative
    df = lambda x: f(x).dx
    
    # Exact derivative, calculated by hand
    df_ref = lambda x: 2 * x


    # Compare at some points
    cs = linspace(-1, 1)
    for c in cs:
        assert df(c) == df_ref(c)


def test_polynomial():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return -x ** 3 + 2 * x**2 - x + (x**2 - 2)**3

    # Automatic derivative
    df = lambda x: f(x).dx

    # Exact derivative, calculated by hand
    df_ref = lambda x: -3*x**2 + 4*x - 1 + 3 * (x**2 - 2)**2 * 2*x

    # Compare at some points
    cs = linspace(-1, 1)
    for c in cs:
        assert df(c) == df_ref(c)

def test_algebraic():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return x**3 / (x**2 + 1)

    # Automatic derivative
    df = lambda x: f(x).dx

    # Exact derivative, calculated by hand
    df_ref = lambda x: (3*x**2 * (x**2 + 1) - (2*x**4)) / (x**2 + 1)**2

    # Compare at some points
    cs = linspace(-1, 1)
    for c in cs:
        assert df(c) == approx(df_ref(c))