import numpy as np
import pandas as pd
from scipy.stats import rankdata


def binary_max(df_1, df_2):
    return df_2.mask(df_1 > df_2, df_1)


def binary_min(df_1, df_2):
    return df_2.mask(df_1 < df_2, df_1)


def log(df: pd.DataFrame):
    """
    Parameters
    ----------
    df

    Returns
    -------

    """
    return np.log(df)


def abs(df: pd.DataFrame):
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    return np.abs(df)


def sign(df: pd.DataFrame):
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    return np.sign(df)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    cross-sectional rank

    Parameters
    ----------
    df

    Returns
    -------

    """
    return df.rank(axis=1, pct=True)


def ts_delay(df: pd.DataFrame, window: int):
    """
    value of df window days ago

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.shift(window)


def ts_corr(df_x: pd.DataFrame, df_y: pd.DataFrame, window: int):
    """
    time-serial correlation of df_x and df_y for the past window days

    Parameters
    ----------
    df_x
    df_y
    window

    Returns
    -------

    """

    return df_x.rolling(window).corr(df_y).replace([np.inf, -np.inf], np.nan)


def ts_cov(df_x: pd.DataFrame, df_y: pd.DataFrame, window: int):
    """
    time-serial covariance of df_x and df_y for the past window days

    Parameters
    ----------
    df_x
    df_y
    window

    Returns
    -------

    """
    return df_x.rolling(window).cov(df_y).replace([np.inf, -np.inf], np.nan)


def scale(df: pd.DataFrame, k: int):
    """

    Parameters
    ----------
    df
    k

    Returns
    -------

    """
    return df.mul(k).div(np.abs(df).sum())


def ts_delta(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.diff(window)


def ts_decay_linear(df: pd.DataFrame, window: int):
    """
    weighted moving average over the past window days with linearly decaying

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    weights = np.array(range(1, window + 1))
    sum_weights = np.sum(weights)
    return df.rolling(window).apply(lambda x: np.sum(weights * x) / sum_weights)


def industry_neutralize(df: pd.DataFrame, g):
    # todo
    pass


def ts_sum(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).sum()


def ts_max(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).max()


def ts_min(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).min()


def ts_argmax(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).apply(np.argmin) + 1


def ts_rank(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """

    def rolling_rank(x):
        return rankdata(x, method='min')[-1]

    return df.rolling(window).apply(rolling_rank)


def ts_product(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).apply(np.prod)


def ts_std(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).std()


def ts_mean(df: pd.DataFrame, window: int):
    """

    Parameters
    ----------
    df
    window

    Returns
    -------

    """
    return df.rolling(window).mean()













