from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).absolute().parents[1] / 'data'


def get_data(filename: str = '^GSPC.csv', interpolate: bool = True) -> pd.DataFrame:
    """
    Imports pre-downloaded data from csv file and fills missing values in the dataset.

    :param filename: name of the file to be loaded
    :param interpolate: if True the missing values will be filled by interpolation
    :return: the dataset
    """
    # the data are manually pre-downloaded as it seems that Yahoo API does not work...
    res = pd.DataFrame.from_csv(DATA_PATH / filename)
    res = res.asfreq('B')  # type: pd.DataFrame

    if interpolate:
        res = res.interpolate()  # type: pd.DataFrame

    # change names of columns to lowercase and no spaces
    res.columns = [col.lower().replace(' ', '_') for col in res.columns]
    return res

# Only for testing ...
if __name__ == '__main__':
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.finance import candlestick_ohlc

    # volume plot
    df = get_data()
    df.volume.plot()

    # candlestick plot
    plt.figure()
    ax1 = plt.subplot()
    ax1.xaxis_date()
    plt.xlabel("Date")

    df['time'] = mdates.date2num(df.index.to_pydatetime())
    df = df[['time'] + list(df.columns)[:-1]]

    candlestick_ohlc(ax1, df.values)

    plt.show()
