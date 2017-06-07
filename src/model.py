"""
Useful link:
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html

How to predict from unrelated dataset with SARIMAX
https://github.com/statsmodels/statsmodels/issues/2577
"""
import pandas as pd
import statsmodels.api as sm

ModelResult = sm.tsa.statespace.MLEResults
SARIMAX = sm.tsa.statespace.SARIMAX


def predict_from_unrelated(new_data: pd.Series, model_result: ModelResult, model: SARIMAX, steps: int = 1) -> pd.Series:
    """
    Predicts out-of-sample data for already-fitted model for data, which the model did not see before.

    Inspiration taken from: https://github.com/statsmodels/statsmodels/issues/2577

    :param new_data: data from which should carried out the prediction
    :param model_result: result of the fitted model
    :param model: fitted model, which should be used for the prediction
    :param steps: number of steps to be predicted
    :return: the prediction
    """

    mod_new = SARIMAX(new_data, order=model.order, seasonal_order=model.seasonal_order, trend=model.trend)
    res_new = mod_new.filter(model_result.params)

    return res_new.forecast(steps=steps)


def reference(time_series_train: pd.Series) -> (ModelResult, SARIMAX):
    """
    Fit reference model to a time series.
    :param time_series_train:
    :return: result and fitted model
    """
    mod = SARIMAX(time_series_train, trend='ct', order=(8, 1, 1))
    res = mod.fit(disp=False)

    return res, mod


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from get_data import get_data
    from preprocessing import adjust_to_seasonality, split_data

    COL = 'adj_volume'

    # get data
    df = get_data()

    # adjusted volume

    df['adj_volume'] = adjust_to_seasonality(df.volume, transformations=['scale', ])

    # ACF and PACF
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df[COL], lags=100, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df[COL], lags=50, ax=ax2)

    plt.figure()

    train_df, test_df = split_data(df, random=False)

    train = train_df[COL]
    test = test_df[COL]

    train.plot()
    test.plot()

    res_, mod_ = reference(train)

    res_.fittedvalues.plot()

    prediction = res_.forecast(steps=2)  # type: pd.Series
    prediction.plot()

    prediction = predict_from_unrelated(test[:10], model_result=res_, model=mod_, steps=2)  # type: pd.Series
    prediction.plot()

    plt.show()
