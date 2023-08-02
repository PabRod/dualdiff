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

    def __neg__(self):
        """ Negation operator """
        y = -self.x
        dy = -self.dx
        return Dual(y, dy)
    
    def __pos__(self):
        """ Positive operator """
        return self
    
    def __sub__(self, other):
        """ Left-side subtraction operator """
        other = Dual(other)  # Coerce into dual
        return self + (-other)
    
    def __rsub__(self, other):
        """ Right-side subtraction operator """
        return other + (-self)
    
    def __mul__(self, other):
        """ Left-side multiplication operator """
        other = Dual(other)  # Coerce into dual
        y = self.x * other.x
        dy = self.dx * other.x + self.x * other.dx # Product rule for derivatives
        return Dual(y, dy)
    
    def __rmul__(self, other):
        """ Right-side multiplication operator """
        return other * self
    
    def __truediv__(self, other):
        """ Left-side division operator """
        other = Dual(other)  # Coerce into dual
        y = self.x / other.x
        dy = (self.dx * other.x - self.x * other.dx) / other.x**2
        return Dual(y, dy)
    
    def __rtruediv__(self, other):
        """ Right-side division operator """
        return other / self


