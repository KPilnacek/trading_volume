import numpy as np
import pandas as pd

from .base import TimeSeries, BaseModel


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
