#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from typing import List

import get_data

__author__ = "Krystof Pilnacek"
__description__ = '''
    Modelling and prediction of S&P 500 daily trading volume.
'''


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    return parser.parse_args(argv[1:])


def run():
    sp_historical = get_data.get_data()


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    run()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
