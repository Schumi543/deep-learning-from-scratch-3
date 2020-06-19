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
