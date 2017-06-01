#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from typing import List

import get_data
from preprocessing import adjust_to_seasonality

__author__ = "Krystof Pilnacek"
__description__ = '''
    Modelling and prediction of S&P 500 daily trading volume.
'''


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-f',
                        '--frequency',
                        type=int,
                        default=90,
                        help='Frequency of seasonal effects (in days)')

    return parser.parse_args(argv[1:])


def run(
        frequency: int = 90,
):
    # load data
    sp = get_data.get_data()

    # preprocess
    sp_volume_adjusted = adjust_to_seasonality(sp.volume, freq=frequency)

    # todo: model...


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    run(
        frequency=args.frequency,
    )

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
