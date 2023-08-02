from dualdiff.dual import Dual
from dualdiff.decorators import autodifferentiable

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
    c = 5
    assert(df(c) == df_ref(c))