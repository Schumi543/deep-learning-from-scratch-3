is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable, Function
    from dezero.core_simple import using_config, no_grad, setup_variable
    from dezero.core_simple import as_array, as_variable
else:
    from dezero.core import Variable, Function
    from dezero.core import using_config, no_grad, setup_variable
    from dezero.core import as_array, as_variable

setup_variable()
