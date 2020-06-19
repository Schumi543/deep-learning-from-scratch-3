import pytest
from dezero import *

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
    y = 3.0 * x + 1.0
    assert isinstance(y, Variable)
    assert y.data == 7


def test_neg():
    x = Variable(np.array(2.0))
    y = -x
    assert y.data == -2


def test_sub():
    x = Variable(np.array(2.0))
    y1 = 2.0 - x
    y2 = x - 1.0

    assert y1.data == 0
    assert y2.data == 1


def test_pow():
    x = Variable(np.array(2.0))
    y = x ** 3

    assert y.data == 8
