import pytest
from step.second_stage import *

import numpy as np


def test_add():
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys: tuple = f(xs)
    assert ys[0].data == 5
