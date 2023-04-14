import pandas as pd
import numpy as np


def mad_method(df: pd.DataFrame, n: float) -> pd.DataFrame:
    """
    中位数去极值

    参数：
    df: pd.DataFrame，输入数据框
    factor: float，中位数绝对偏差乘数，默认为3.0

    返回：
    去除离群点后的数据框
    """
    median = df.median()  # 计算中位数
    mad = df.mad()  # 计算中位数绝对偏差
    threshold = n * mad  # 计算阈值
    df[df > median + threshold] = np.nan  # 大于阈值的值赋为nan
    df[df < median - threshold] = np.nan  # 小于阈值的值赋为nan
    df.fillna(median, inplace=True)  # 缺失值填充为中位数
    return df


def market_neutralize(factor_df: pd.DataFrame, market_cap_df: pd.DataFrame) -> pd.DataFrame:
    """
    市值中性化

    参数：
    factor_df: pd.DataFrame，输入数据框，包含股票因子值
    market_cap_df: pd.DataFrame，股票市值数据框

    返回：
    去除市场因素后的因子值
    """
    # 计算股票收益率的协方差矩阵和市值权重
    return_df = factor_df.pct_change().dropna()
    cov_matrix = return_df.cov()
    market_cap_weight = market_cap_df.div(market_cap_df.sum(axis=1), axis=0)

    # 计算市场收益率和股票对市场收益率的敏感性（即beta值）
    market_return = return_df.mean(axis=1)
    beta = cov_matrix.apply(lambda x: x / x.sum()) @ market_cap_weight.T

    # 计算市值中性化后的因子值
    market_neutral_factor = factor_df.sub(beta.mul(market_return.sub(return_df.mean(axis=1)), axis=0), axis=0)

    return market_neutral_factor


def norm_fn(df):
    return (df - df.mean())/df.std()

