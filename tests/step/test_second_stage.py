import pytest
from step.second_stage import *

import numpy as np


def test_add():
    assert add(Variable(np.array(2)), Variable(np.array(3))).data == 5


def test_simple_backward():
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = add(square(x), square(y))
    z.backward()

    assert z.data == 13
    assert x.grad == 4
    assert y.grad == 6


def test_gradient_step14():
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()

    assert y.data == 6
    assert x.grad == 2
    x.clear_grad()  # teardown 1st tests

    y = add(add(x, x), x)
    y.backward()

    assert x.grad == 3


def test_complex_backward():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    assert y.data == 32
    assert x.grad == 64


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
