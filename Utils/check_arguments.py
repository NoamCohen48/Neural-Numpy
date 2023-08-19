import functools
import inspect
from typing import Any, Callable


def check_arguments(func: Callable) -> Callable:
    """
    checks arguments type and return type real time
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(func)

        bind = sig.bind(*args, **kwargs)
        bind.apply_defaults()
        parameters = sig.parameters

        for parameter_name, parameter_object in parameters.items():
            if parameter_object.annotation is parameter_object.empty:
                continue
            parameter_value = bind.arguments[parameter_name]
            parameter_should_type: type = parameter_object.annotation
            assert isinstance(parameter_value, parameter_should_type), \
                (f"parameter {parameter_name} should have type of {parameter_should_type.__name__}, "
                 f"but passed {parameter_value} of type {type(parameter_value).__name__}")

        result = func(*args, **kwargs)
        return_should_type: type = sig.return_annotation

        if return_should_type is sig.empty:
            return result
        else:
            assert isinstance(result, return_should_type), \
                (f"return value {result} should have type of {return_should_type.__name__}, "
                 f"but has type of {type(result).__name__}")
            return result

    return wrapper
