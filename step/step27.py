import numpy as np
from dezero import Variable
from dezero.math_functions import sin
from dezero.utils import plot_dot_graph

if __name__ == '__main__':
    x = Variable(np.array(np.pi / 4))
    y = sin(x)
    y.backward()

    x.name = 'x'
    y.name = 'y'

    plot_dot_graph(y, verbose=False, to_file='sin.png')

    y = sin(x, threshold=1e-150)
    y.backward()

    plot_dot_graph(y, verbose=False, to_file='sin_1e-150.png')
