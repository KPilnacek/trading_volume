"""
Useful link:
http://www.statsmodels.org/stable/statespace.html
http://www.statsmodels.org/stable/examples/notebooks/generated/statespace_local_linear_trend.html
"""


import pandas as pd
import datetime as dt
import numpy as np
import _tools
from model.base import BaseModel, TimeSeries


class MVSimple(BaseModel):

    def __init__(self, n_t: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_t = n_t

        # parameter initialization
        self.t_par = np.arange(self.n_t) * .1 + .1  # type: np.ndarray
        self.v_par = np.arange(self._train.shape[1]) * .1 + .1  # type: np.ndarray

        self._fitted_values = None

    def time_comp(self, time: np.ndarray) -> np.ndarray:
        assert time.shape == self.t_par.shape, 'Time and its params must have the same shape'
        return (time * self.t_par).reshape(1, -1)

    def indep_comp(self, y: np.ndarray) -> np.ndarray:
        return y * self.v_par[0]

    def dep_comp(self, x_mat: np.ndarray) -> np.ndarray:
        if len(x_mat.shape) == 1:
            return x_mat * self.v_par[1:]
        else:
            return x_mat @ self.v_par[1:]

    def first_diff(self, time: np.ndarray, var: np.ndarray) -> float:

        assert var.shape[0] == len(time), 'Number of variable rows must be the same as time length'
        assert var.shape[1] == len(self.v_par), 'Number of variable cols must be the same as number of its params'

        return self.time_comp(time) @ (self.indep_comp(var[:, 0]) + self.dep_comp(var[:, 1:]))

    def step(self, t_p: pd.Timestamp, data: pd.DataFrame, ) -> pd.DataFrame:
        time = (data.index - t_p).components.days.values

        one_step = (0 - time[-1]) * self.first_diff(time, data.values) + data.iloc[-1, 0]

        return pd.DataFrame(one_step, index=[t_p], columns=[self._train.columns[0]])

    def forecast(self, steps: int = 1, **kwargs):
        index = pd.bdate_range(self._train.index[-1], periods=steps + 1)[1:]

        return self.step(index[-1], self._train.iloc[-self.n_t:])

    @property
    @_tools.drop_undefined
    def resid(self) -> pd.DataFrame:
        return self.fitted_values - self._train.iloc[:, 0]

    @property
    def fitted_values(self) -> pd.Series:
        if self._fitted_values is None:
            _fitted_values = self.rolling_forecast(new_data=self._train)
            self._fitted_values = _fitted_values
        else:
            _fitted_values = self._fitted_values

        return _fitted_values

    def forecast_from_unrelated(self, new_data: TimeSeries, steps: int = 1, **kwargs) -> pd.DataFrame:
        index = pd.bdate_range(new_data.index[-1], periods=steps + 1)[1:]
        return self.step(index[-1], new_data.iloc[-self.n_t:])

if __name__ == '__main__':

    n_t = 10
    n_x = 5
    t = np.arange(1, n_t + 1)
    p = np.arange(1, n_t + 1)
    x_p = np.arange(1, n_x + 1)
    X = np.tile(t, (n_x, 1)).T

    data_ = pd.DataFrame(np.hstack((t.reshape(-1, 1), X)),
                         index=pd.bdate_range(start=dt.datetime(2017, 6, 9), periods=n_t),
                         columns=['y'] + ['x' + str(i) for i in range(n_x)])

    print(MVSimple(time_series_train=data_, n_t=n_t).forecast(1))
