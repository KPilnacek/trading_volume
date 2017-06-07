"""
Useful link:
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html

How to predict on out-of-sample data with SARIMAX
https://github.com/statsmodels/statsmodels/issues/2577
"""
import pandas as pd
import statsmodels.api as sm


def reference(time_series_train: pd.Series) -> sm.tsa.statespace.MLEResults:
    """
    Fit reference model to a time series.
    :param time_series_train:
    :return:
    """
    mod = sm.tsa.statespace.SARIMAX(time_series_train, trend='ct', order=(10, 1, 1))
    res = mod.fit(disp=False)
    print(res.summary())

    return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from get_data import get_data
    from preprocessing import adjust_to_seasonality, split_data

    COL = 'adj_volume'

    # get data
    df = get_data()

    # adjusted volume
    factor = - np.floor(max(np.log10(df.volume)))
    df['adj_volume'] = adjust_to_seasonality(df.volume, transformations=['scale', ], factor=10**factor)

    # ACF and PACF
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df[COL], lags=100, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df[COL], lags=50, ax=ax2)

    plt.figure()

    A, B = split_data(df, random=False)

    train = A[COL]
    test = B[COL]

    train.plot()
    test.plot()

    res_ = reference(train)

    res_.fittedvalues.plot()

    prediction = res_.forecast(steps=10)  # type: pd.Series
    prediction.plot()

    plt.show()
