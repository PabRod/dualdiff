from dualdiff.dual import Dual

def autodifferentiable(f):
    """ Autodifferentiable decorator 
    
    Seamlessly turns a given function 
    f = lambda x: expr(x) 
    into
    f = lambda x: expr(Dual(x, 1)

    How to use it:

    @autodifferentiable
    def f(x : Number):
        ...
    """
    def decorated(x):
        return f(Dual(x, 1))
    
    return decorated
