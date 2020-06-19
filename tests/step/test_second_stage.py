import pytest
from step.second_stage import *

import numpy as np


def test_add():
    assert add(Variable(np.array(2)), Variable(np.array(3))).data == 5
