# Dualdiff

(Forward) automatic differentiation from scratch

## Description

The purpose of this project is learning about automatic differentiation.
There are many Python libraries about this topic which certainly are more efficient and professional than this one.
Nevertheless, I wanted to implement it from scratch in order to get a deep understanding of it.
And boy! The mathematics behind it are beautiful!

## Dual numbers

Forward automatic differentiation is made possible by [dual numbers](https://en.wikipedia.org/wiki/Dual_number).
Dual numbers are similar to two-dimensional vectors:

$$
z \equiv (u, u')
$$

and the idea is that $u$ represents the value of a function at a given point $f(x_0)$ and $u'$ the derivative at the same point $f'(x_0)$

Dual numbers can be summed, subtracted and multipliced by a scalar following the usual linearity rules:

$$
\alpha (u, u') + \beta (v, v') \equiv (\alpha u + \beta v, \alpha u' + \beta v')
$$

But they have special rules for multiplication and division.
Multiplication, for instance, looks like:

$$
(u, u') \cdot (v, v') \equiv (u v, u'v + u v')
$$

Looks familiar?
The first "coordinate" just implements a regular multiplication.
While the second "coordinate" describes the rule of product for derivatives.

Regarding divison, the idea is the same:

$$
\frac{(u, u')}{(v, v')} \equiv (\frac{u}{v}, \frac{u'v - uv'}{v^2})
$$

It is also quite useful to define the power of a dual number and a real number as:

$$
(u, u')^n = (u^n, n \cdot u^{n-1} \cdot u')
$$

Dual numbers defined like this have an extraordinary property:

> If $q(x)$ is **any** algebraic function[^1]
>
> then
>
> $q((x_0, 1)) = (q(x_0), q'(x_0))$

This means that just evaluating the function with $(x_0, 1)$ instead of $x_0$ returns the derivative!

I implemented these basic properties (and others) in the class `Dual`.

---

But hey, what about non-algebraic functions[^2]?
Just as we'll do with an undergraduate math student, we can teach our dual numbers the derivative of some basic functions.
Remember those tables you had to learn in high school?

For instance, if we define:

$$
\sin (u, u') \equiv (\sin u, \cos u \cdot u')
$$

now our dual numbers can also deal with functions involving sines.

I've implemented some of these derivatives in `dualdiff/primitives.py`.
The list can be easily extended in case of need.

## How to use

Imagine you want to differentiate the function:

```python
from numpy import sin

def f(x):
    return sin(x ** 2)
```

All you have to do is to use the `@autodifferentiable` decorator:

```python
from dualdiff.primitives import *
from dualdiff.decorators import autodifferentiable

@autodifferentiable
def f(x):
    return sin(x ** 2) + x
```

Decorated this way, by evaluating $f$ we get its value and it's derivative:

```python
x0 = 3
f(x0)

> Dual(3.4121184852417565, # f(x_0)
       -4.466781571308061) # f'(x_0)
```

[^1]: that is, a function only involving sums, subtractions, multiplications, divisons and powers.

[^2]: such as, for instance, anything involving a cosine.