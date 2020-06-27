import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as f

iterss = 0

if __name__ == '__main__':

    x = Variable(np.array(1))
    y = f.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)

    for iters in [0, 1, 2, 3, 4, 5, 6]:
        for i in range(iters):
            gx: Variable = x.grad
            x.clear_grad()
            gx.backward(create_graph=True)

        gx: Variable = x.grad
        gx.name = f'gx{iters + 1}'
        plot_dot_graph(gx, verbose=False, to_file=f'tanh_diff_{iters + 1}.png')
