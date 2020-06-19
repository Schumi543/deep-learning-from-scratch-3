import pytest
from step.first_stage import *

import numpy as np


def test_composite_function():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    y = C(B(A(x)))

    assert y.data == pytest.approx(1.64872)


def test_numerical_diff():
    f = Square()
    x = Variable(np.array(2.0))

    assert numerical_diff(f, x) == pytest.approx(4.00000)


def test_numerical_diff_composite():
    A = Square()
    B = Exp()
    C = Square()

    f = lambda x: C(B(A(x)))
    x = Variable(np.array(0.5))

    assert numerical_diff(f, x) == pytest.approx(3.29744)


def test_backward():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    assert x.grad == pytest.approx(3.29744)


def test_backprop():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.arg == b
    assert y.creator.arg.creator == B
    assert y.creator.arg.creator.arg == a
    assert y.creator.arg.creator.arg.creator == A
    assert y.creator.arg.creator.arg.creator.arg == x
