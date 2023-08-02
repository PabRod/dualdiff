from dualdiff.dual import Dual

def test_init():
    z = Dual(2, 1)
    assert z.x == 2 
    assert z.dx == 1

    z = Dual(1)
    assert z.x == 1
    assert z.dx == 0

    u = Dual(z)
    assert z.x == 1
    assert z.dx == 0

def test_eq():
    u = Dual(2, 1)
    v = Dual(2, 1)
    w = Dual(2, 0)
    q = Dual(1, 1)

    assert u == v
    assert u != w
    assert u != q