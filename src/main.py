#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from typing import List

import _tools
import get_data
import preprocessing

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

    parser.add_argument('-p',
                        '--percent',
                        type=_tools.parse_num_in_range,
                        default=.8,
                        help='Fraction of provided data to be used to train model.')

    return parser.parse_args(argv[1:])


def run(
        frequency: int = 90,
        frac_train: float = .8,
):
    # load data
    sp = get_data.get_data()

    # preprocess
    sp['adj_volume'] = preprocessing.adjust_to_seasonality(sp.volume, freq=frequency, transformations=['decompose_resid'])
    train, test = preprocessing.divide_data(sp, frac_train=frac_train)

    # todo: model...


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    run(
        frequency=args.frequency,
        frac_train=args.percent,
    )

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
