import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
from dezero.math_functions import goldstein


if __name__ == '__main__':
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    # noinspection PyTypeChecker
    z = goldstein(x, y)
    z.backward()

    x.name = 'x'
    y.name = 'y'
    z.name = 'z'

    plot_dot_graph(z, verbose=False, to_file='goldstein.png')
