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
