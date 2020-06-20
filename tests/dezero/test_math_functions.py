import pytest
from dezero import Variable
from dezero.math_functions import sphere, matyas, goldstein, sin, rosenbrock

import numpy as np


def test_sphere():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = sphere(x, y)
    z.backward()

    assert (x.grad, y.grad) == (2, 2)


def test_matyas():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = matyas(x, y)
    z.backward()

    assert x.grad == pytest.approx(0.04)
    assert y.grad == pytest.approx(0.04)


def test_goldstein():
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


def test_rosenbrock():
    x0 = Variable(np.array(0))
    x1 = Variable(np.array(2))

    y = rosenbrock(x0, x1)
    y.backward()

    assert (x0.grad, x1.grad) == (-2, 400)
