"""
Useful link:
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
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
    from get_data import get_data
    from preprocessing import adjust_to_seasonality, split_data
    import matplotlib.pyplot as plt

    # get data
    df = get_data()

    # adjusted volume
    df['adj_volume'] = adjust_to_seasonality(df.volume, transformations=['decompose_resid'])

    A, B = split_data(df, random=False)
    A.volume.plot()
    B.volume.plot()

    # ACF and PACF
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df.volume, lags=50, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df.volume, lags=50, ax=ax2)

    res_ = reference(A.volume)
    res_.fittedvalues.plot()
    # res.predict(0, 925)
    res_.forecast(steps=10)
    # df.adj_volume.plot()
    plt.show()
