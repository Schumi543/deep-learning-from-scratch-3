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