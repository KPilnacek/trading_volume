"""
Preprocessing of time series from stock markets.

Useful links:

Seasonality analysis
====================

A Simple Time Series Analysis Of The S&P 500 Index
http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/

Statistical forecasting: notes on regression and time series analysis (Duke UNI)
http://people.duke.edu/~rnau/411home.htm

What model to use:
http://people.duke.edu/~rnau/whatuse.htm (the same as before)

Seasonal ARIMA with Python
http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/

A comprehensive beginner’s guide to create a Time Series Forecast (with Codes in Python)
https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
"""
from typing import Optional, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


def drop_undefined(func):
    """
    Drops all undefined values (infinity, NaN).
    Decorator for function, which returns pd.Series.

    :param func: Function to decorate (should return pd.Series)
    :return: Decorated function
    """
    def wrapper(*args, **kwargs) -> pd.Series:
        original_res = func(*args, **kwargs)  # type: pd.Series
        assert isinstance(original_res, pd.Series), 'The decorated function should return pd.Series'

        no_nans = original_res.dropna()
        finite = no_nans[np.isfinite(no_nans)]

        return finite
    return wrapper


@drop_undefined
def transform(transformation: str, ts: pd.Series) -> pd.Series:
    """
    Mathematical transformation of time series using various types of transformations.

    :param transformation: Type of transformation
    :param ts: Time series, which should be transformed
    :return: Transformed time series
    """
    transformation_dict = {
        'log': lambda x: np.log(x),
        'first_diff': lambda x: (x - x.shift(1)).dropna()
    }

    if transformation not in transformation_dict:
        raise NotImplementedError(f'{transformation.capitalize()} is not implemented.')

    return transformation_dict.get(transformation, lambda x: x)(ts)


def adjust_to_seasonality(time_series: pd.Series,
                          freq: int = 90,
                          transformations: Optional[List[str]] = None,
                          ) -> pd.Series:
    """
    Removes seasonal pattern and trend from the data in order to allow modelling of the residuals.
    Allows several mathematical transformations.

    :param time_series: Time series, which should be transformed
    :param freq: Frequency of the seasonal pattern (in days)
    :param transformations: The mathematical transformation to be applied.
    :return: Transformed time series
    """

    # mathematical transformation
    if transformations is not None:
        for transformation in transformations:
            time_series = transform(transformation, time_series)

    # seasonal decomposition
    res = sm.tsa.seasonal_decompose(time_series, freq=freq)

    return res.resid

if __name__ == '__main__':
    from get_data import get_data
    import matplotlib.pyplot as plt

    # get data
    df = get_data()

    # show different adjustments to seasonality
    transformations_ = [['log'], ['first_diff', 'log'], ['log', 'first_diff']]
    for tr in transformations_:
        adjust_to_seasonality(df.volume, transformations=tr).plot()
    plt.legend([str(tr) for tr in transformations_], loc='best')
    plt.title('Adjustment to seasonality with different transformations')

    # show seasonal decomposition of volume with different frequencies
    df['first_diff'] = df.volume - df.volume.shift(1)
    df.dropna(inplace=True)

    for freq_ in range(30, 120, 30):
        res_ = sm.tsa.seasonal_decompose(np.log(df.volume), freq=freq_)
        resplot = res_.plot()
        resplot.suptitle(f'Analyzed with frequency {freq_} days')

    plt.show()
