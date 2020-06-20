from dezero.core_simple import Variable, Function


def _dot_var(v: Variable, verbose=False):
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += f"{str(v.shape)} {str(v.dtype)}"
    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'


def _dot_func(f: Function):
    txt = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'

    for x in f.args:
        txt += f'{id(x)} -> {id(f)}\n'
    for y in f.outputs:
        txt += f'{id(f)} -> {id(y())}\n'

    return txt
