from dualdiff.dual import Dual
from dualdiff.primitives import *
from dualdiff.decorators import autodifferentiable
import numpy as np
from pytest import approx

def test_sin():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return sin(x)
    
    # Automatic derivative
    df = lambda x: f(x).dx
    
    # Exact derivative, calculated by hand
    df_ref = lambda x: np.cos(x)

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == df_ref(c)


def test_cos():

    @autodifferentiable
    def f(x):
        """ Function enabled for automatic differentiation """
        return cos(x)

    # Automatic derivative
    def df(x): return f(x).dx

    # Exact derivative, calculated by hand
    def df_ref(x): return -np.sin(x)

    # Compare at some points
    cs = np.linspace(-1, 1)
    for c in cs:
        assert df(c) == df_ref(c)


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
