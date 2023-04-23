import pandas as pd
import numpy as np


def alpha_pre_process(alpha):
    def src_norm(x):
        return (x - x.mean())/x.std()
    alpha["y_pred"] = alpha["y_pred"].unstack().apply(src_norm, axis=1).stack()
    return alpha


def alpha_get_position_and_turnover(alpha):
    def src_long_short_balance(x):
        x_ = x.copy()
        x_positive = x[x > 0]
        x_negative = x[x < 0]
        x_[x > 0] = x_positive / x_positive.sum()
        x_[x < 0] = -x_negative / x_negative.sum()
        return x_/2
    alpha["position"] = alpha["y_pred"].unstack().apply(src_long_short_balance, axis=1).stack()
    alpha["turnover"] = alpha["position"].unstack().sort_index().diff().abs().stack()
    return alpha


def cal_information_coefficient(alpha, mode="spearman"):
    ic = alpha.groupby("date").apply(lambda x: x.corr(mode).loc["y_pred", "y"])
    return ic


def cal_pnl(alpha):
    ret = alpha.groupby("date").apply(lambda x: np.nansum(x["position"] * x["ret"]))
    return ret


def cal_win_rate(ret):
    return (ret > 0).sum() / len(ret)


def cal_max_drawdown(ret):
    cum_ret = (1 + ret).cumprod()
    cum_max = cum_ret.cummax()
    drawdown = cum_ret / cum_max - 1
    max_drawdown = drawdown.min()
    return max_drawdown


def create_full_tear_sheet(alpha):
    alpha = alpha_pre_process(alpha)
    alpha = alpha_get_position_and_turnover(alpha)
    ic = cal_information_coefficient(alpha, "pearson")
    rank_ic = cal_information_coefficient(alpha, "spearman")
    pnl_daily = cal_pnl(alpha)

    win_rate = cal_win_rate(pnl_daily)
    max_drawdown = cal_max_drawdown(pnl_daily)
    return {"IC": ic.mean(),
            "Rank IC": rank_ic.mean(),
            "ARR": pnl_daily.mean() * 242,
            "AV": pnl_daily.std() * np.sqrt(242),
            "Sharpe": pnl_daily.mean() / pnl_daily.std() * np.sqrt(242),
            "WR": win_rate,
            "MDD": max_drawdown
            }
