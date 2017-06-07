"""
Useful link:
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html

How to predict from unrelated dataset with SARIMAX
https://github.com/statsmodels/statsmodels/issues/2577
"""
from typing import Type, Union

import pandas as pd
import statsmodels.api as sm

ModelResult = sm.tsa.statespace.MLEResults
ModelClass = sm.tsa.statespace.MLEModel
SARIMAX = sm.tsa.SARIMAX
VARMAX = sm.tsa.VARMAX


class Model(object):
    """
    Holds model of time series from `statsmodels` package

    :param time_series_train: time series on which the model should be trained
    :param model: `stasmodels` timeseries model (*e*. *g*. SARIMAX, VARMAX)
    :param kwargs: parameters for the models
    """

    def __init__(
            self,
            time_series_train: Union[pd.Series, pd.DataFrame],
            model: Type[ModelClass] = SARIMAX,
            **kwargs
    ):

        self._model = model(endog=time_series_train, **kwargs)
        self._kwargs = kwargs

        self._model_result = self._model.fit(disp=False)

    def forecast_from_unrelated(self, new_data: pd.Series, steps: int = 1, **kwargs) -> pd.Series:
        """
        Predicts out-of-sample data for already-fitted model for data, which the model did not see before.

        Inspiration taken from: https://github.com/statsmodels/statsmodels/issues/2577

        :param new_data: data from which should carried out the prediction
        :param steps: number of steps to forecast
        :return: forecast values
        """

        mod_new = type(self._model)(endog=new_data, **self._kwargs)
        res_new = mod_new.filter(self._model_result.params)

        return res_new.forecast(steps=steps, **kwargs)

    def forecast(self, steps: int = 1, **kwargs) -> pd.Series:
        """
        Forecasts out-of-sample data based on the train data.

        :param steps: number of steps to forecast
        :param kwargs:
        :return: forecast values
        """
        res = self._model_result.forecast(steps, **kwargs)  # type: pd.Series
        return res

    @property
    def fitted_values(self) -> pd.Series:
        """The predicted values of the model."""
        return self._model_result.fittedvalues


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

    reference = Model(train, model=SARIMAX, trend='ct', order=(8, 1, 1))

    reference.fitted_values.plot()
    reference.forecast(steps=2).plot()
    reference.forecast_from_unrelated(test[:10], steps=2).plot()

    plt.show()
