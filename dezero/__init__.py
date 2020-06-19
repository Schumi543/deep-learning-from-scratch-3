is_simple_core = True

if is_simple_core:
    from dezero.core_simple import Variable, Function
    from dezero.core_simple import using_config, no_grad, setup_variable
    from dezero.core_simple import as_array, as_variable

setup_variable()
