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
        self.x = z.x
        self.dx = z.dx

    def __str__(self):
        return "Dual({0}, {1})".format(self.x, self.dx)

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        """ Equality operator """
        other = Dual(other)  # Coerce into dual
        return all([self.x == other.x, self.dx == other.dx])
    
    def __neq__(self, other):
        """ Inequality operator """
        return not self == other
    
    def __add__(self, other):
        """ Addition operator """
        other = Dual(other) # Coerce into dual
        y = self.x + other.x
        dy = self.dx + other.dx
        return Dual(y, dy)
