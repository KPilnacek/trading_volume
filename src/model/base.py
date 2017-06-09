import abc
import datetime as dt
import warnings
from pathlib import Path
from typing import NamedTuple, Optional, Union

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('pdf')
from matplotlib import pyplot as plt

TimeSeries = Union[pd.Series, pd.DataFrame]

SAVE_FIG_PATH = Path(__file__).absolute().parents[2] / 'plots'


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
            column: Optional[str] = None,
            **kwargs
    ):
        self._train = time_series_train
        self._test = time_series_test

        self.column = column
        if self.column is None:
            self._train_plot = self._train  # type: pd.Series
            self._test_plot = self._test  # type: pd.Series
        else:
            self._train_plot = self._train[self.column]
            if self._test is not None:
                self._test_plot = self._test[self.column]

        self._kwargs = kwargs

    @abc.abstractmethod
    def forecast_from_unrelated(self, new_data: TimeSeries, steps: int = 1, **kwargs) -> pd.Series:
        """
        Predicts out-of-sample data for already-fitted model for data, which the model did not see before.

        :param new_data: data from which should carried out the prediction
        :param steps: number of steps to forecast
        :return: forecast values
        """
        raise NotImplementedError

    def rolling_forecast(self, new_data: TimeSeries, lag: int = 10, ) -> pd.Series:
        """
        Uses method *forecast_from_unrelated* for rolling prediction over long interval.

        :param new_data: test data
        :param lag: number of previous steps to take into account during prediction
        :return: prediction
        """
        res = []
        idx = new_data.index
        for i in range(1, len(new_data)):
            start = max(i - lag, 0)
            if self.column is None:
                data = new_data[idx[start:i]]
            else:
                data = new_data.iloc[start:i]

            res.append(self.forecast_from_unrelated(data, steps=1))
        res_pd = pd.concat(res)  # type: pd.Series
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

    @property
    @abc.abstractclassmethod
    def _impulse_responses(self) -> pd.Series:
        """
        10 steps of response to an impulse
        :return: the response
        """
        raise NotImplementedError

    def plot_impulse_response(self, plot_args: dict, save_plots: bool = False):
        """
        Plots 10 steps of response function
          
        :param plot_args: arguments for plotting (title, ylabel) 
        :param save_plots: if `True` the plots are saved
        """

        plt.figure(plot_args.get('title', str(self)) + ' impulse response')

        self._impulse_responses.plot()

        plt.ylabel(plot_args.get('ylabel', '').capitalize() + ' ' + (self.column or self._train.name))
        plt.legend(['impulse response'], loc='best')

        if save_plots:
            SAVE_FIG_PATH.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(SAVE_FIG_PATH / (plot_args.get('title', str(self)) + '_impulse_resp' + '.pdf')))

    def results(self,
                steps: Optional[int] = None,
                lag: int = 10,
                show_plots: bool = False,
                to_print: bool = False,
                save_plots: bool = False,
                plot_args: Optional[dict] = None,
                ) -> Results:
        """
        Plots train and test (if provided) data and returns simple statistics object.

        :param steps: number steps to be predicted on the test data
        :param lag: number of steps to predict from
        :param show_plots: if `True` the plots are shown
        :param to_print:  if `True` the simple statistics are printed
        :param save_plots:  if `True` the figures are saved
        :param plot_args: arguments for `matplotlib` (title, ylabel) 
        :return: Simple statistics on test data (SSE, R^2)
        """
        if plot_args is None:
            plot_args = {'title': str(self) + dt.datetime.now().strftime('.%M%S')}

        if self.column is None:
            sse = (self.resid ** 2).sum()
        else:
            sse = (self.resid[self.column] ** 2).sum()

        sst = ((self._train_plot - self._train_plot.mean()) ** 2).sum()
        train_res = Results(sse=sse, sst=sst)

        plt.figure(plot_args.get('title', str(self)))

        self._train_plot.plot()
        if self.column is None:
            self.fitted_values.plot()
        else:
            self.fitted_values[self.column].plot()

        if self._test is not None:

            if steps is None:
                n_test_samples = len(self._test_plot)
            else:
                n_test_samples = min(len(self._test_plot), steps)

            self._test_plot.plot()

            rolling_forecast = self.rolling_forecast(self._test[:n_test_samples], lag=lag)
            if self.column is not None:
                rolling_forecast = rolling_forecast[self.column]
            rolling_forecast.plot()

            sse_u = ((self._test_plot[:n_test_samples] - rolling_forecast).dropna() ** 2).sum()
            sst_u = ((self._test_plot[rolling_forecast.index]
                      - self._test_plot[rolling_forecast.index].mean()) ** 2).sum()

            unrel_res = Results(sse_u, sst_u)
        else:
            warnings.warn('Test data not provided.')
            unrel_res = Results()

        if to_print:
            self.print(train_res, unrel_res)

        plt.ylabel(plot_args.get('ylabel', '').capitalize() + ' ' + (self.column or self._train.name))
        plt.legend(['train data', 'predicted train data', 'test data', 'predicted test data'], loc='best')

        if save_plots:
            SAVE_FIG_PATH.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(SAVE_FIG_PATH / (plot_args.get('title', str(self)) + '.pdf')))

        self.plot_impulse_response(plot_args, save_plots)

        if show_plots:
            plt.show()

        return unrel_res
