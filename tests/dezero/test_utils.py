import numpy as np

from dezero import Variable
# noinspection PyProtectedMember
from dezero.utils import _dot_var, _dot_func


def test__dot_var():
    x = Variable(np.random.randn(2, 3))
    x.name = 'x'
    assert _dot_var(x) == f'{id(x)} [label="x", color=orange, style=filled]\n'
    assert _dot_var(x, verbose=True) == f'{id(x)} [label="x: (2, 3) float64", color=orange, style=filled]\n'


def test__dot_func():
    x0 = Variable(np.array(1))
    x1 = Variable(np.array(1))
    y = x0 + x1
    txt = _dot_func(y.creator)

    assert txt == f'{id(y.creator)} [label="Add", color=lightblue, style=filled, shape=box]\n' \
                  f'{id(x0)} -> {id(y.creator)}\n' \
                  f'{id(x1)} -> {id(y.creator)}\n' \
                  f'{id(y.creator)} -> {id(y)}\n'
