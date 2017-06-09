#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from pprint import pprint
from typing import List

import _tools
import get_data
import model
import preprocessing

__author__ = "Krystof Pilnacek"
__description__ = '''
    Modelling and prediction of S&P 500 daily trading volume.
'''


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-f',
                        '--file',
                        type=str,
                        default='^GSPC.tsv',
                        help='Filename of time series to be analyzed')

    parser.add_argument('--fraction',
                        type=_tools.parse_num_in_range,
                        default=.8,
                        help='Fraction of provided data to be used to train model.')

    return parser.parse_args(argv[1:])


def run(
        filename: str = '^GSPC.csv',
        frac_train: float = .8,
):
    # load data
    sp = get_data.get_data(filename=filename)

    # preprocess
    df_scaled = sp.apply(preprocessing.adjust_to_seasonality, args=(['scale', ],))
    df_first_diff = sp.apply(preprocessing.adjust_to_seasonality, args=(['first_diff', ],))

    res = {}

    for transform, df in {'scaled': df_scaled, 'first_diff': df_first_diff}.items():
        train, test = preprocessing.split_data(df, frac_train=frac_train)

        # reference model
        reference = model.reference.Reference(train['volume'], test['volume'])
        res['reference; ' + transform] = reference.results(show_plots=False)

        # univariate model
        sarimax = model.statespace_models.Model(train['volume'],
                                                test['volume'],
                                                model=model.statespace_models.SARIMAX,
                                                trend='ct', order=(4, 1, 4), enforce_invertibility=False)
        res['sarimax; ' + transform] = sarimax.results(show_plots=False)

        # multivariate model
        varmax = model.statespace_models.Model(train[['open', 'close', 'volume']],
                                               test[['open', 'close', 'volume']],
                                               column='volume',
                                               model=model.statespace_models.VARMAX,
                                               trend='c', order=(4, 1))
        res['varmax; ' + transform] = varmax.results(show_plots=True)

    pprint(res)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    run(
        filename=args.file,
        frac_train=args.fraction,
    )

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
