import pytest
from dezero import Variable
from dezero.math_functions import sphere, matyas, goldstein, sin

import numpy as np


def test_add():
    assert (Variable(np.array(2)) + Variable(np.array(3))).data == 5


def test_gradient_step14():
    x = Variable(np.array(3.0))
    y = x + x
    y.backward()

    assert y.data == 6
    assert x.grad == 2
    x.clear_grad()  # teardown 1st tests

    y = (x + x) + x
    y.backward()

    assert x.grad == 3


def test_variable_add():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    y = a + b

    assert y.data == 5


def test_variable_mul():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    y = a * b

    assert y.data == 6


def test_as_variable():
    y = Variable(np.array(2.0)) + np.array(3.0)
    assert isinstance(y, Variable)
    assert y.data == 5


def test_ops_with_float():
    x = Variable(np.array(2.0))
    # noinspection PyTypeChecker
    y = 3.0 * x + 1.0
    assert isinstance(y, Variable)
    assert y.data == 7


def test_neg():
    x = Variable(np.array(2.0))
    y = -x
    assert y.data == -2


# noinspection PyTypeChecker
def test_sub():
    x = Variable(np.array(2.0))
    y1 = 2.0 - x
    y2 = x - 1.0

    assert y1.data == 0
    assert y2.data == 1


def test_pow():
    x = Variable(np.array(2.0))
    # noinspection PyTypeChecker
    y = x ** 3

    assert y.data == 8


def test_sphere_function():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = sphere(x, y)
    z.backward()

    assert (x.grad, y.grad) == (2, 2)


def test_matyas_function():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = matyas(x, y)
    z.backward()

    assert x.grad == pytest.approx(0.04)
    assert y.grad == pytest.approx(0.04)


def test_goldstein_function():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = goldstein(x, y)
    z.backward()

    assert x.grad == -5376
    assert y.grad == 8064


def test_sin():
    x = Variable(np.array(np.pi / 4))
    y = sin(x)
    y.backward()

    assert y.data == pytest.approx(0.707106)
    assert x.grad == pytest.approx(0.707103)
