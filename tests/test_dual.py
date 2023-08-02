from dualdiff.dual import Dual

def test_init():
    z = Dual(2, 1)

    assert(z.x == 2)
    assert(z.dx == 1)