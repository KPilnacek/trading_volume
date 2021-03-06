import argparse
import functools
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAVE_FIG_PATH = Path(__file__).absolute().parents[1] / 'plots'


def is_number(number: Any) -> bool:
    """
    Check if the given argument can be converted to `float` or `int`.

    :param number: Would-be number
    :return: True if `number` can be converted to `float` or `int`.
    """
    for type_ in (float, int):
        try:
            type_(number)
        except (TypeError, ValueError) as e:
            continue
        else:
            return True
    return False


def parse_num_in_range(number_str: str, start: float = 0, end: float = 1) -> float:
    """
    Argparse type for parsing number in range between `start` and `end` values. If the `number_str` is not in the given
    range ArgumentTypeError is raised.

    :param number_str: Number as string to parse
    :param start: Start of the range
    :param end: End of the range
    :return: Floating number of the parsed number
    """

    if not is_number(number_str):
        msg = "Argument '{}' is not a number.".format(str(number_str))
        raise argparse.ArgumentTypeError(msg)
    number = float(number_str)

    if not (start <= number <= end):
        msg = "Argument '{}' is not in the specified range [{}, {}].".format(str(number_str), str(start), str(end))
        raise argparse.ArgumentTypeError(msg)

    return number


def drop_undefined(func):
    """
    Drops all undefined values (infinity, NaN).
    Decorator for function, which returns pd.Series.

    :param func: Function to decorate (should return pd.Series)
    :return: Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> pd.Series:
        original_res = func(*args, **kwargs)  # type: pd.Series

        assert isinstance(original_res, pd.Series), 'The decorated function should return pd.Series'

        no_nans = original_res.dropna()
        finite = no_nans[np.isfinite(no_nans)]

        return finite
    return wrapper


def save_fig(title: str, figname: Optional[Union[str, int, float]] = None):
    plt.figure((figname or title))

    SAVE_FIG_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(SAVE_FIG_PATH / (title + '.pdf')), format='pdf')

# only for testing...
if __name__ == '__main__':
    for s in ['a', '.1', '50', {}, set(), 'bla', '1.001']:
        print(s, is_number(s))
