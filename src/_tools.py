import argparse
from typing import Any


def is_number(number: Any) -> bool:
    """
    Check if the given argument can be converted to `float` or `int`.

    :param number: Would-be number
    :return:
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


# only for testing...
if __name__ == '__main__':
    for s in ['a', '.1', '50', {}, set(), 'bla', '1.001']:
        print(s, is_number(s))
