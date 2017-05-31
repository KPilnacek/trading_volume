from pathlib import Path

import pandas as pd

SP_DATA = Path(__file__).absolute().parents[1] / 'data' / '^GSPC.csv'


def get_data() -> pd.DataFrame:
    res = pd.DataFrame.from_csv(SP_DATA)
    res.columns = [col.lower().replace(' ', '_') for col in res.columns]
    return res


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
