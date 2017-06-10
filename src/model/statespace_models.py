"""
Useful link:
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
http://vmm.math.uci.edu/ODEandCM/PDF_Files/Appendices/AppendixI.pdf

How to predict from unrelated dataset with SARIMAX
https://github.com/statsmodels/statsmodels/issues/2577
"""

from typing import Optional, Type

import pandas as pd
from statsmodels import api as sm

from model.base import BaseModel, TimeSeries

ModelResult = sm.tsa.statespace.MLEResults
ModelClass = sm.tsa.statespace.MLEModel

SARIMAX = sm.tsa.SARIMAX
ARIMA = sm.tsa.ARIMA
VARMAX = sm.tsa.VARMAX


class Model(BaseModel):
    """
    Holds model of time series from `statsmodels` package

    :param time_series_train: time series on which the model should be trained
    :param time_series_test: time series on which the model should be tested
    :param model: `stasmodels` timeseries model (*e*. *g*. SARIMAX, VARMAX)
    :param kwargs: parameters for the models
    """

    def __init__(
            self,
            time_series_train: TimeSeries,
            time_series_test: Optional[TimeSeries] = None,
            model: Type[ModelClass] = SARIMAX,
            **kwargs
    ):

        super(Model, self).__init__(time_series_train, time_series_test, **kwargs)

        self._model = model(endog=self._train, **kwargs)
        self._model_result = self._model.fit(disp=False)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._model.__class__.__name__})'

    def forecast_from_unrelated(self, new_data: TimeSeries, steps: int = 1, **kwargs) -> pd.Series:
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
        res = self._model_result.forecast(steps, **kwargs)  # type: pd.Series
        return res

    @property
    def fitted_values(self) -> pd.Series:
        return self._model_result.fittedvalues

    @property
    def resid(self) -> pd.Series:
        return self._model_result.resid

    @property
    def _impulse_responses(self):
        if self.column is None:
            return self._model_result.impulse_responses(10, orthogonalized=True)
        else:
            return self._model_result.impulse_responses(10, orthogonalized=True)[self.column]

    def print(self, *args, **kwargs):
        print(self._model_result.summary())
        super().print(*args, **kwargs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from get_data import get_data
    from preprocessing import adjust_to_seasonality
    from _tools import save_fig

    df = get_data()
    volume = adjust_to_seasonality(df.volume, ['scale', ])

    # ACF and PACF
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_acf(volume, lags=100, ax=ax1)
    fig = plt.figure(figsize=(12, 4))
    ax2 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(volume, lags=50, ax=ax2)
    save_fig('acf', 1)
    save_fig('pacf', 2)