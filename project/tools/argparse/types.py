"""Module contains necessary argparse module extensions or functions."""


from functools import partial
from typing import Callable


def positive_type(value: str, cast_f: Callable, include_zero: bool = False):
    """
    Wrap type for ``argparse``.

    This function is a type for ``argparse`` package argument.
    This type has additional limitation on range of values - only positive.

    Parameters
    ----------
    value : str
        Value that will be processed by type-cast.
    cast_f: Callable
        Function that makes base-cast. Typical values are: `int`, `float`.
        Casted type should support simple comparison.
    include_zero : bool, optional
        Bool flag that controls zero include in OK-range.

    Returns
    -------
    int:
        Casted to the int argument.

    Raises
    ------
    ValueError:
        This exception will be raised if casted value is not positive.
    """
    new_value = cast_f(value)
    if include_zero:
        check_f = lambda x: x >= 0
    else:
        check_f = lambda x: x > 0
    if not check_f(new_value):
        raise IOError(f'Argument that must be positive has negative '
                      f'value: "{new_value}".')
    return new_value


natural_int = partial(positive_type, cast_f=int)  # noqa
natural_float = partial(positive_type, cast_f=float)  # noqa
non_negative_int = partial(positive_type, cast_f=int, include_zero=True)  # noqa
non_negative_float = partial(positive_type, cast_f=float, include_zero=True)  # noqa


def unit_interval(value: str):
    """
    Wrap type for ``argparse``. This type defines float in ``[0..1]``.

    This function is a type for ``argparse`` package argument..

    Parameters
    ----------
    value : str
        Value that will be processed by type-cast.

    Returns
    -------
    int:
        Casted to the int argument.

    Raises
    ------
    ValueError:
        This exception will be raised if casted value is not positive.
    """
    new_value = float(value)
    if new_value < 0 or new_value > 1:
        raise IOError(f'Argument that must be in [0..1] has wrong '
                      f'value: "{new_value}".')
    return new_value
