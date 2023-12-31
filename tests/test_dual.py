from dualdiff.dual import Dual
from numpy import log

def test_init():
    z = Dual(2, 1)
    assert z.x == 2 
    assert z.dx == 1

    q = Dual(1)
    assert q.x == 1
    assert q.dx == 0

    u = Dual(z)
    assert u.x == 2
    assert u.dx == 1

def test_eq():
    u = Dual(2, 1)
    v = Dual(2, 1)
    w = Dual(2, 0)
    q = Dual(1, 1)

    assert u == v
    assert u != w
    assert u != q

def test_sum():
    u = Dual(2, 1)
    v = Dual(1, 0)
    y = Dual(3, 1)

    assert y == u + v
    assert y == v + u

def test_neg():
    u = -Dual(2, 1)
    y = Dual(-2, -1)

    assert y == u

def test_pos():
    u = +Dual(2, 1)
    y = Dual(2, 1)

    assert y == u

def test_sub():    
    u = Dual(2, 1)
    v = Dual(1, 0)

    assert u - v == Dual(1, 1)
    assert v - u == Dual(-1, -1)

def test_mul():    
    u = Dual(2, 1)
    v = Dual(3, 0)
    y = Dual(6, 3)

    assert u * v == y
    assert v * u == y

def test_div():
    u = Dual(4, 1)
    v = Dual(2, 2)

    assert u / v == Dual(4 / 2, 
                         (2*1 - 2*4)/2**2)
    
    assert v / u == Dual(2 / 4, 
                         (4*2 - 2)/4**2)
    
def test_pow():
    u = Dual(4, 1)
    y = Dual(64, 3*4**2)

    assert u ** 3 == y

def test_rpow():
    u = Dual(3, 1)
    y = Dual(8, log(2) * 2 ** 3)

    assert 2 ** u == y

def test_abs():
    p = Dual(3, 1)
    n = Dual(-2, 0)

    assert abs(p) == 3
    assert abs(n) == 2