"""
Useful link:
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html

How to predict from unrelated dataset with SARIMAX
https://github.com/statsmodels/statsmodels/issues/2577
"""

from typing import Optional, Type

import pandas as pd
from statsmodels import api as sm

from .base import BaseModel, TimeSeries

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

    def print(self, *args, **kwargs):
        print(self._model_result.summary())
        super().print(*args, **kwargs)