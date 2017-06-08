"""
Useful link:
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html

How to predict from unrelated dataset with SARIMAX
https://github.com/statsmodels/statsmodels/issues/2577
"""
import abc
import warnings
from typing import Type, Union, Optional, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ModelResult = sm.tsa.statespace.MLEResults
ModelClass = sm.tsa.statespace.MLEModel
TimeSeries = Union[pd.Series, pd.DataFrame]

SARIMAX = sm.tsa.SARIMAX
ARIMA = sm.tsa.ARIMA
VARMAX = sm.tsa.VARMAX


class Results(NamedTuple):
    """
    Represents resulting statistics

    :param sse: Sum of squares of errors
    :param sst: Total sum of squares
    """
    sse: float = np.nan
    sst: float = np.nan

    def __repr__(self):
        return f'{self.__class__.__name__}(SSE={self.sse:6.2f}, SST={self.sst:6.2f}, R^2={self.r2:6.4f})'

    @property
    def r2(self) -> float:
        """
        R squared computed from SSE and SST.
        :return: R^2
        """
        return 1-self.sse/self.sst


class BaseModel(object, metaclass=abc.ABCMeta):
    """
    Abstract class of time series model

    :param time_series_train: time series on which the model should be trained
    :param time_series_test: time series on which the model should be tested
    :param kwargs: parameters for the models
    """

    def __init__(
            self,
            time_series_train: TimeSeries,
            time_series_test: Optional[TimeSeries] = None,
            **kwargs
    ):
        self._train = time_series_train
        self._test = time_series_test

        self._kwargs = kwargs

    @abc.abstractmethod
    def forecast_from_unrelated(self, new_data: TimeSeries, steps: int = 1, **kwargs) -> pd.Series:
        """
        Predicts out-of-sample data for already-fitted model for data, which the model did not see before.

        Inspiration taken from: https://github.com/statsmodels/statsmodels/issues/2577

        :param new_data: data from which should carried out the prediction
        :param steps: number of steps to forecast
        :return: forecast values
        """
        raise NotImplementedError

    def rolling_forecast(self, new_data: TimeSeries, lag: int = 10, ) -> TimeSeries:
        """
        Uses method *forecast_from_unrelated* for rolling prediction over long interval.

        :param new_data: test data
        :param lag: number of previous steps to take into account during prediction
        :return: prediction
        """
        res = []
        idx = new_data.index
        for i in range(lag, len(new_data)):
            res.append(self.forecast_from_unrelated(new_data[idx[i-lag:i]], steps=1))
        res_pd = pd.concat(res)  # type: TimeSeries
        return res_pd

    @abc.abstractmethod
    def forecast(self, steps: int = 1, **kwargs) -> pd.Series:
        """
        Forecasts out-of-sample data based on the train data.

        :param steps: number of steps to forecast
        :param kwargs:
        :return: forecast values
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fitted_values(self) -> pd.Series:
        """The predicted values of the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def resid(self) -> pd.Series:
        """The model residuals"""
        raise NotImplementedError

    def print(self, train_stat: Results, unrelated_stat: Results, ):
        """
        Prints simple statistics of the model.

        :param train_stat: Statistics of train data
        :param unrelated_stat: Statistics of unrelated data
        """
        print(f'{self}, SSE: {train_stat.sse:7.2f},' + ' ' * 11
              + f'R^2: {train_stat.r2:7.4f},\n'
              + ' ' * (len(str(self)) - 8)
              + f'unrelated SSE: {unrelated_stat.sse:7.2f}, unrelated R^2: {unrelated_stat.r2:7.4f}')

    def results(self, steps: Optional[int] = None, lag: int = 10, show_plots: bool = False, ) -> Results:
        """
        Plots train and test (if provided) data and returns simple statistics object.

        :param steps: number steps to be predicted on the test data
        :param lag: number of steps to predict from
        :param show_plots: if `True` the plots are shown
        :return: Simple statistics (SSE, R^2)
        """
        sse = (self.resid ** 2).sum()
        sst = ((self._train - self._train.mean()) ** 2).sum()
        train_res = Results(sse=sse, sst=sst)

        plt.figure(str(self))

        self._train.plot()
        self.fitted_values.plot()

        if self._test is not None:

            if steps is None:
                n_test_samples = len(self._test)
            else:
                n_test_samples = min(len(self._test), steps + lag)

            self._test.plot()

            rolling_forecast = self.rolling_forecast(self._test[:n_test_samples], lag=lag)
            rolling_forecast.plot()

            sse_u = ((self._test[:n_test_samples] - rolling_forecast).dropna() ** 2).sum()
            sst_u = ((self._test[rolling_forecast.index] - self._test[rolling_forecast.index].mean()) ** 2).sum()

            unrel_res = Results(sse_u, sst_u)
        else:
            warnings.warn('Test data not provided.')
            unrel_res = Results()

        self.print(train_res, unrel_res)

        if show_plots:
            plt.show()

        return unrel_res


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


class Reference(BaseModel):
    """
    Persistence model
    """

    def __repr__(self):
        return f'Model({self.__class__.__name__})'

    @property
    def fitted_values(self) -> pd.Series:
        return self._train.shift(1)

    def forecast(self, steps: int = 1, **kwargs) -> pd.Series:
        index = pd.bdate_range(self._train.index[-1], periods=steps + 1)[1:]
        return pd.Series([self._train.values[-1]]*steps, index=index)

    def forecast_from_unrelated(self, new_data: TimeSeries, steps: int = 1, **kwargs) -> pd.Series:
        index = pd.bdate_range(new_data.index[-1], periods=steps + 1)[1:]
        return pd.Series([new_data.values[-1]] * steps, index=index)

    @property
    def resid(self) -> pd.Series:
        res = self._train - self.fitted_values
        res[res.isnull()] = res[np.where(res.isnull())[0]+1].values
        return res


if __name__ == '__main__':
    from get_data import get_data
    from preprocessing import adjust_to_seasonality, split_data

    COL = 'volume'

    # get data
    df = get_data()

    # adjusted volume
    # df['adj_volume'] = adjust_to_seasonality(df.volume, transformations=['scale', ])
    df = df.apply(adjust_to_seasonality, args=(['scale', ],))

    # ACF and PACF
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df[COL], lags=100, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df[COL], lags=50, ax=ax2)

    train_df, test_df = split_data(df, random=False)

    train = train_df[COL]
    test = test_df[COL]

    ref = Reference(train, test)
    ref.results()

    sarimax = Model(train, test, model=SARIMAX, trend='c', order=(4, 1, 4))
    # varmax = Model(train_df[['open', 'high', 'low', 'adj_close', 'volume']], model=VARMAX, trend='c', order=(8, 1))
    sarimax.results()

    plt.show()
