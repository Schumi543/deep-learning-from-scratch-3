import pytest

import logging
import numpy as np

from dezero import Variable
from dezero.functions import sphere, matyas, goldstein, taylor_sin, rosenbrock, sin

LOGGER = logging.getLogger(__name__)


def test_sphere():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = sphere(x, y)
    z.backward()

    assert (x.grad.data, y.grad.data) == (2, 2)


def test_matyas():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = matyas(x, y)
    z.backward()

    assert x.grad.data == pytest.approx(0.04)
    assert y.grad.data == pytest.approx(0.04)


def test_goldstein():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = goldstein(x, y)
    z.backward()

    assert x.grad.data == -5376
    assert y.grad.data == 8064


def test_taylor_sin():
    x = Variable(np.array(np.pi / 4))
    y = taylor_sin(x)
    y.backward()

    assert y.data == pytest.approx(0.707106)
    assert x.grad.data == pytest.approx(0.707103)


def test_rosenbrock():
    x0 = Variable(np.array(0))
    x1 = Variable(np.array(2))

    y = rosenbrock(x0, x1)
    y.backward()

    assert (x0.grad.data, x1.grad.data) == (-2, 400)


def test_higher_derivative_sin():
    x = Variable(np.array(1))
    y = sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        gx = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)
        LOGGER.debug('{} {}'.format(i, x.grad))

    assert x.grad.data == pytest.approx(0.8414709)
