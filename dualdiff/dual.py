from plum import dispatch
from numbers import Number

class Dual:

    @dispatch
    def __init__(self, x: Number, dx: Number = 0):
        """ Basic constructor """
        self.x = x
        self.dx = dx

    @dispatch
    def __init__(self, z: "Dual"):
        """ Additional constructor
        If the input is already a dual number, just return it.
        This will be handy later. """
        self = z

    