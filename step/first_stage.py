import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, arg: Variable):
        x = arg.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4):
    y0 = f(Variable(x.data - eps))
    y1 = f(Variable(x.data + eps))

    return (y1.data - y0.data) / (2 * eps)
