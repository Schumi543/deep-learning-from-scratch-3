import pytest
from step.first_stage import *

import numpy as np


@pytest.fixture(scope="module")
def setup_graph():
    A = Square()
    B = Exp()
    C = Square()

    def f(arg):
        return C(B(A(arg)))

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    return A, B, C, x, a, b, y, f


def test_composite_function(setup_graph):
    A, B, C, x, a, b, y, f = setup_graph

    assert y.data == pytest.approx(1.64872)


def test_numerical_diff():
    f = Square()
    x = Variable(np.array(2.0))

    assert numerical_diff(f, x) == pytest.approx(4.00000)


def test_numerical_diff_composite(setup_graph):
    A, B, C, x, a, b, y, f = setup_graph

    assert numerical_diff(f, x) == pytest.approx(3.29744)


def test_backward(setup_graph):
    A, B, C, x, a, b, y, f = setup_graph

    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    assert x.grad == pytest.approx(3.29744)


def test_backprop(setup_graph):
    A, B, C, x, a, b, y, f = setup_graph

    assert y.creator == C
    assert y.creator.arg == b
    assert y.creator.arg.creator == B
    assert y.creator.arg.creator.arg == a
    assert y.creator.arg.creator.arg.creator == A
    assert y.creator.arg.creator.arg.creator.arg == x


def test_backprop_2(setup_graph):
    _, _, _, _, _, _, y, _ = setup_graph

    C = y.creator
    b = C.arg
    b.grad = C.backward(y.grad)

    B = b.creator
    a = B.arg
    a.grad = B.backward(b.grad)

    A = a.creator
    x = A.arg
    x.grad = A.backward(a.grad)

    assert x.grad == pytest.approx(3.29744)


def test_backprop_3(setup_graph):
    _, _, _, x, _, _, y, _ = setup_graph

    y.backward()
    assert x.grad == pytest.approx(3.29744)


def test_variable_type():
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        Variable(1.0)
