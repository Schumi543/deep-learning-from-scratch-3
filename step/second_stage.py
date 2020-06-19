import numpy as np
from typing import List
import weakref


class Variable:
    # noinspection PyTypeChecker
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data: np.ndarray = data
        self.grad: np.ndarray = None
        self.creator: Function = None
        self.generation: int = 0

    # noinspection PyAttributeOutsideInit
    def set_creator(self, f):
        self.creator = f
        self.generation = f.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        # FIXME priority queue may make it simpler
        def add_func(f: Function):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.args, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def clear_grad(self):
        self.grad = None


class Function:
    def __call__(self, *args):
        xs = [x.data for x in args]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(_as_array(y)) for y in ys]

        # update generation
        self.generation = max([x.generation for x in args])

        # memorize
        for output in outputs:
            output.set_creator(self)
        self.args = args
        self.outputs = [weakref.ref(output) for output in outputs]

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


def _as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
