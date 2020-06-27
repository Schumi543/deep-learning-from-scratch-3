import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as f

if __name__ == '__main__':

    x = Variable(np.linspace(-7, 7, 200))
    y = f.sin(x)
    y.backward(create_graph=True)

    logs = [y.data.flatten()]

    for i in range(3):
        logs.append(x.grad.data.flatten())
        gx = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)

    labels = ["y=sin(x", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc='lower right')
    plt.show()
