from dualdiff.dual import Dual

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
