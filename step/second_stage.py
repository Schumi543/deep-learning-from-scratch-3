import numpy as np
from typing import List


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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.args, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *args):
        xs = [x.data for x in args]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(_as_array(y)) for y in ys]

        # memorize
        for output in outputs:
            output.set_creator(self)
        self.args = args
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


class Add(Function):
    def forward(self, *xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.args[0].data
        return 2 * x * gy

def square(x):
    return Square()(x)

def numerical_diff(f, x, eps=1e-4):
    y0 = f(Variable(_as_array(x.data - eps)))
    y1 = f(Variable(_as_array(x.data + eps)))

    return (y1.data - y0.data) / (2 * eps)


def _as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
