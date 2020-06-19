import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data: np.ndarray = data
        self.grad: np.ndarray = None
        self.creator: Function = None

    # noinspection PyAttributeOutsideInit
    def set_creator(self, f):
        self.creator = f

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.arg, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, arg: Variable):
        x = arg.data
        y = self.forward(x)
        output = Variable(_as_array(y))

        # memorize
        output.set_creator(self)
        self.arg = arg
        self.output = output

        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError

    @staticmethod
    def _as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.arg.data
        return 2 * x * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.arg.data
        return np.exp(x) * gy


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def numerical_diff(f, x, eps=1e-4):
    y0 = f(Variable(x.data - eps))
    y1 = f(Variable(x.data + eps))

    return (y1.data - y0.data) / (2 * eps)
