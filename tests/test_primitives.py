from dualdiff.dual import Dual
from dualdiff.primitives import *
from dualdiff.decorators import autodifferentiable
from pytest import approx

def _test_factory(f, df_ref, cs=np.linspace(-1, 1), use_approx=False):
    """ Auxilary testing function

    f: the function to be automatically differentiated
    df_ref: exact derivative, calculated by hand
    cs: points to sample and compare
    use_approx: set to True for approximate comparisons

    The function compares f(x) and df_ref(x) at multiple points
    """
    # Function enabled for automatic differentiation
    f = autodifferentiable(f)

    # Automatic derivative
    def df(x): return f(x).dx

    # Compare at given points
    for c in cs:
        if use_approx:
            assert df(c) == approx(df_ref(c))
        else:
            assert df(c) == df_ref(c)

def test_sin():
    _test_factory(lambda x: sin(x) + sin(1),
                  lambda x: cos(x))


def test_cos():
    _test_factory(lambda x: cos(x) - cos(np.pi),
                  lambda x: -sin(x))


def test_composition():
    _test_factory(lambda x: cos(sin(x**2)),
                  lambda x: -sin(sin(x**2)) * cos(x**2) * 2 * x,
                  use_approx=True)

def test_tan():
    _test_factory(lambda x: tan(x),
                  lambda x: 1/cos(x)**2,
                  use_approx=True)


def test_exp():
    _test_factory(lambda x: exp(3*x),
                  lambda x: 3*exp(3*x),
                  use_approx=True)

def test_funpow():
    _test_factory(lambda x: exp(x)**cos(x),
                  lambda x: exp(x * cos(x)) * (cos(x) - x * sin(x)),
                  use_approx=True)

def test_verbose():

    def f(x):
        """ Just a code-intense way of expressing a polynomial of 3rd degree"""
        v = 0
        for n in [0, 1, 2, 3]:
            v += x ** n

        return v
    
    _test_factory(f,
                  lambda x: 3 * x**2 + 2 * x + 1,
                  use_approx=True)

def test_piecewise():

    def f(x):
        if x < 0:
            return x ** 2
        else:
            return x ** 3

    # Exact derivative, calculated by hand
    def df_ref(x): 
        if x < 0:
            return 2 * x
        else:
            return 3 * x ** 2

    _test_factory(f,
                  df_ref,
                  use_approx=True)
