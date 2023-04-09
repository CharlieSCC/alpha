from alpha.factor.alpha import Alphas
from alpha.factor.operators import *


class Alphas101(Alphas):

    @staticmethod
    def alpha_001(underlying):
        # Alpha#1	 rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5
        close = underlying.data["close"].copy()
        ret = close.pct_change()
        close[ret < 0] = ts_std(ret, 20)
        return cs_rank(ts_argmax(close ** 2 * sign(close), 5)) - 0.5

    @staticmethod
    # Alpha#2	 (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha002(underlying):
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        opn = underlying.data["open"].copy()
        df = -1 * ts_corr(cs_rank(ts_delta(log(volume), 2)), cs_rank((close - opn) / opn), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    @staticmethod
    def alpha_003(underlying):
        # Alpha#3	 (-1 * correlation(rank(open), rank(volume), 10))
        opn = underlying.data["open"].copy()
        volume = underlying.data["volume"].copy()
        res = -1 * ts_corr(cs_rank(opn), cs_rank(volume), 10)
        return res.replace([-np.inf, np.inf], 0).fillna(value=0)

    @staticmethod
    def alpha_004(underlying):
        # Alpha#4	 (-1 * Ts_Rank(rank(low), 9))
        close = underlying.data["close"].copy()
        return -1 * ts_rank(cs_rank(close), 9)

    @staticmethod
    def alpha_005(underlying):
        # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
        vwap = underlying.data["vwap"].copy()
        opn = underlying.data["open"].copy()
        close = underlying.data["close"].copy()
        return cs_rank((opn - (ts_sum(vwap, 10) / 10))) * (-1 * abs(cs_rank((close - vwap))))

    @staticmethod
    def alpha_006(underlying):
        # Alpha#6	 (-1 * correlation(open, volume, 10))
        opn = underlying.data["open"].copy()
        volume = underlying.data["volume"].copy()
        df = -1 * ts_corr(opn, volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    @staticmethod
    def alpha_007(underlying):
        # Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        adv20 = ts_mean(volume, 20)
        alpha = -1 * ts_rank(abs(ts_delta(close, 7)), 60) * sign(ts_delta(close, 7))
        alpha[adv20 >= volume] = -1
        return alpha

    @staticmethod
    def alpha_008(underlying):
        # Alpha#8	 (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
        opn = underlying.data["open"].copy()
        close = underlying.data["close"].copy()
        ret = close.pct_change()
        return -1 * (cs_rank(((ts_sum(opn, 5) * ts_sum(ret, 5)) - ts_delay((ts_sum(opn, 5) * ts_sum(ret, 5)), 10))))

    @staticmethod
    def alpha_009(underlying):
        # Alpha#9	 ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
        close = underlying.data["close"].copy()
        delta_close = ts_delta(close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    @staticmethod
    def alpha_010(underlying):
        # Alpha#10	 rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
        close = underlying.data["close"].copy()
        delta_close = ts_delta(close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return cs_rank(alpha)

    @staticmethod
    def alpha_011(underlying):
        # Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
        vwap = underlying.data["vwap"].copy()
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        return (cs_rank(ts_max((vwap - close), 3)) + cs_rank(ts_min((vwap - close), 3))) * cs_rank(ts_delta(volume, 3))

    @staticmethod
    def alpha_012(underlying):
        # Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        return sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))

    @staticmethod
    def alpha_013(underlying):
        # Alpha#13	 (-1 * rank(covariance(rank(close), rank(volume), 5)))
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        return -1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))

    @staticmethod
    def alpha_014(underlying):
        # Alpha#14	 ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        opn = underlying.data["open"].copy()
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        ret = close.pct_change()
        df = ts_corr(opn, volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * cs_rank(ts_delta(ret, 3)) * df

    @staticmethod
    def alpha_015(underlying):
        # Alpha#15	 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        high = underlying.data["high"].copy()
        volume = underlying.data["volume"].copy()
        df = ts_corr(cs_rank(high), cs_rank(volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(cs_rank(df), 3)

    @staticmethod
    def alpha_016(underlying):
        # Alpha#16	 (-1 * rank(covariance(rank(high), rank(volume), 5)))
        high = underlying.data["high"].copy()
        volume = underlying.data["volume"].copy()
        return -1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), 5))

    @staticmethod
    def alpha_017(underlying):
        # Alpha#17	 (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        adv20 = ts_mean(volume, 20)
        return -1 * (cs_rank(ts_rank(close, 10)) *
                     cs_rank(ts_delta(ts_delta(close, 1), 1)) *
                     cs_rank(ts_rank((volume / adv20), 5)))

    @staticmethod
    def alpha_018(underlying):
        # Alpha#18	 (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
        close = underlying.data["close"].copy()
        opn = underlying.data["open"].copy()
        df = ts_corr(close, opn, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (cs_rank((ts_std(abs((close - opn)), 5) + (close - opn)) + df))

    @staticmethod
    def alpha_019(underlying):
        # Alpha#19	 ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
        close = underlying.data["close"].copy()
        ret = close.pct_change()
        return (-1 * sign((close - ts_delay(close, 7)) + ts_delta(close, 7))) * (1 + cs_rank(1 + ts_sum(ret, 250)))

    @staticmethod
    def alpha_020(underlying):
        # Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
        opn = underlying.data["open"].copy()
        high = underlying.data["high"].copy()
        close = underlying.data["close"].copy()
        low = underlying.data["low"].copy()
        return -1 * (cs_rank(opn - ts_delay(high, 1)) *
                     cs_rank(opn - ts_delay(close, 1)) *
                     cs_rank(opn - ts_delay(low, 1)))

    @staticmethod
    def alpha_021(underlying):
        # Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        cond_1 = ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
        cond_2 = ts_mean(close, 2) < ts_mean(close, 8) - ts_std(close, 8)
        cond_3 = ts_mean(volume, 20) / volume < 1
        return (cond_1 | ((~cond_1) & (~cond_2) & (~cond_3))).astype('int') * (-2) + 1

    @staticmethod
    def alpha_022(underlying):
        # Alpha#22	 (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
        high = underlying.data["high"].copy()
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        df = ts_corr(high, volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_delta(df, 5) * cs_rank(ts_std(close, 20))

    @staticmethod
    def alpha_023(underlying):
        # Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        close = underlying.data["close"].copy()
        high = underlying.data["high"].copy()
        cond = ts_mean(high, 20) < high
        alpha = close.copy(
            deep=True)  # pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=['close'])
        alpha[cond] = -1 * ts_delta(high, 2).fillna(value=0)
        alpha[~cond] = 0
        return alpha

    @staticmethod
    def alpha_024(underlying):
        # Alpha#24	 ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
        close = underlying.data["close"].copy()
        cond = ts_delta(ts_mean(close, 100), 100) / ts_delay(close, 100) <= 0.05
        alpha = -1 * ts_delta(close, 3)
        alpha[cond] = -1 * (close - ts_min(close, 100))
        return alpha

    @staticmethod
    def alpha_025(underlying):
        # Alpha#25	 rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        vwap = underlying.data["vwap"].copy()
        high = underlying.data["high"].copy()
        ret = close.pct_change()
        adv20 = ts_mean(volume, 20)
        return cs_rank(((((-1 * ret) * adv20) * vwap) * (high - close)))

    @staticmethod
    def alpha_026(underlying):
        # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        volume = underlying.data["volume"].copy()
        high = underlying.data["high"].copy()
        df = ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)

    @staticmethod
    def alpha_027(underlying):
        # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        alpha = cs_rank((ts_mean(ts_corr(cs_rank(volume), cs_rank(vwap), 6), 2) / 2.0))
        return sign((alpha - 0.5) * (-2))

    @staticmethod
    def alpha_028(underlying):
        # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        volume = underlying.data["volume"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        close = underlying.data["close"].copy()
        adv20 = ts_mean(volume, 20)
        df = ts_corr(adv20, low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((high + low) / 2)) - close), 1)

    @staticmethod
    def alpha_029(underlying):
        # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        close = underlying.data["close"].copy()
        ret = close.pct_change()
        return ts_min(
            cs_rank(cs_rank(scale(log(ts_sum(cs_rank(cs_rank(-1 * cs_rank(ts_rank((close - 1), 5)))), 2)), 1))),
            5) + ts_rank(ts_delay((-1 * ret), 6), 5)

    @staticmethod
    def alpha_030(underlying):
        # Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        delta_close = ts_delta(close, 1)
        inner = sign(delta_close) + sign(ts_delay(delta_close, 1)) + sign(ts_delay(delta_close, 2))
        return ((1.0 - cs_rank(inner)) * ts_sum(volume, 5)) / ts_sum(volume, 20)

    @staticmethod
    def alpha_031(underlying):
        # Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        volume = underlying.data["volume"].copy()
        low = underlying.data["low"].copy()
        close = underlying.data["close"].copy()
        adv20 = ts_mean(volume, 20)
        df = ts_corr(adv20, low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = cs_rank(cs_rank(cs_rank(ts_decay_linear((-1 * cs_rank(cs_rank(ts_delta(close, 10)))), 10))))
        p2 = cs_rank((-1 * ts_delta(close, 3)))
        p3 = sign(scale(df, 1))
        return p1 + p2 + p3

    @staticmethod
    def alpha_032(underlying):
        # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
        close = underlying.data["close"].copy()
        vwap = underlying.data["vwap"].copy()
        return scale(((ts_mean(close, 7) / 7) - close), 1) + (20 * scale(ts_corr(vwap, ts_delay(close, 5), 230), 1))

    @staticmethod
    def alpha_033(underlying):
        # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
        opn = underlying.data["open"].copy()
        close = underlying.data["close"].copy()
        return cs_rank((-1 + (opn / close)))

    @staticmethod
    def alpha_034(underlying):
        # Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        close = underlying.data["close"].copy()
        ret = close.pct_change()
        inner = ts_std(ret, 2) / ts_std(ret, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return cs_rank(2 - cs_rank(inner) - cs_rank(ts_delta(close, 1)))

    @staticmethod
    def alpha_035(underlying):
        # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        high = underlying.data["high"].copy()
        low = underlying.data["low"].copy()
        ret = close.pct_change()
        return (ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16))) * (1 - ts_rank(ret, 32))

    @staticmethod
    def alpha_036(underlying):
        # Alpha#36	 (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))        adv20 = sma(self.volume, 20)
        close = underlying.data["close"].copy()
        opn = underlying.data["opn"].copy()
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        adv20 = ts_mean(volume, 20)
        ret = close.pct_change()
        return ((((2.21 * cs_rank(ts_corr((close - opn), ts_delay(volume, 1), 15))) + (0.7 * cs_rank((opn - close)))) +
                 (0.73 * cs_rank(ts_rank(ts_delay((-1 * ret), 6), 5)))) + cs_rank(abs(ts_corr(vwap, adv20, 6)))) + \
               (0.6 * cs_rank((((ts_mean(close, 200) / 200) - opn) * (close - opn))))

    @staticmethod
    def alpha_037(underlying):
        # Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        opn = underlying.data["open"].copy()
        close = underlying.data["close"].copy()
        return cs_rank(ts_corr(ts_delay(opn - close, 1), close, 200)) + cs_rank(opn - close)

    @staticmethod
    def alpha_038(underlying):
        # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        close = underlying.data["close"].copy()
        opn = underlying.data["open"].copy()
        inner = close / opn
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * cs_rank(ts_rank(opn, 10)) * cs_rank(inner)

    @staticmethod
    def alpha_039(underlying):
        # Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        adv20 = ts_mean(volume, 20)
        ret = close.pct_change()
        return -1 * cs_rank((ts_delta(close, 7) * (1 - cs_rank(ts_decay_linear((volume / adv20), 9))))) * (
                    1 + cs_rank(ts_mean(ret, 250)))

    @staticmethod
    def alpha_040(underlying):
        # Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        high = underlying.data["high"].copy()
        volume = underlying.data["volume"].copy()
        return -1 * cs_rank(ts_std(high, 10)) * ts_corr(high, volume, 10)

    @staticmethod
    def alpha_041(underlying):
        # Alpha#41	 (((high * low)^0.5) - vwap)
        high = underlying.data["high"].copy()
        low = underlying.data["low"].copy()
        vwap = underlying.data["vwap"].copy()
        return pow((high * low), 0.5) - vwap

    @staticmethod
    def alpha_042(underlying):
        # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
        vwap = underlying.data["vwap"].copy()
        close = underlying.data["close"].copy()
        return cs_rank((vwap - close)) / cs_rank((vwap + close))

    @staticmethod
    def alpha_043(underlying):
        # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        adv20 = ts_mean(volume, 20)
        return ts_rank(volume / adv20, 20) * ts_rank((-1 * ts_delta(close, 7)), 8)

    @staticmethod
    def alpha_044(underlying):
        # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
        high = underlying.data["high"].copy()
        volume = underlying.data["volume"].copy()
        df = ts_corr(high, cs_rank(volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    @staticmethod
    def alpha_045(underlying):
        # Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        df = ts_corr(close, volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (cs_rank(ts_mean(ts_delay(close, 5), 20)) * df * cs_rank(
            ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))

    @staticmethod
    def alpha_046(underlying):
        # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
        close = underlying.data["close"].copy()
        inner = ((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10)
        alpha = -1 * ts_delta(close, 1)
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    @staticmethod
    def alpha_047(underlying):
        # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
        volume = underlying.data["volume"].copy()
        high = underlying.data["high"].copy()
        close = underlying.data["close"].copy()
        vwap = underlying.data["vwap"].copy()
        adv20 = ts_mean(volume, 20)
        return ((((cs_rank((1 / close)) * volume) / adv20) * (
                    (high * cs_rank((high - close))) / (ts_mean(high, 5) / 5))) -
                cs_rank((vwap - ts_delay(vwap, 5))))

    # # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))

    @staticmethod
    def alpha_049(underlying):
        # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        close = underlying.data["close"].copy()
        inner = (((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10))
        alpha = -1 * ts_delta(close, 1)
        alpha[inner < -0.1] = 1
        return alpha

    @staticmethod
    def alpha_050(underlying):
        # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        vwap = underlying.data["vwap"].copy()
        volume = underlying.data["volume"].copy()
        return -1 * ts_max(cs_rank(ts_corr(cs_rank(volume), cs_rank(vwap), 5)), 5)

    @staticmethod
    def alpha_051(underlying):
        # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        close = underlying.data["close"].copy()
        inner = (((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10))
        alpha = -1 * ts_delta(close, 1)
        alpha[inner < -0.05] = 1
        return alpha

    @staticmethod
    def alpha_052(underlying):
        # Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
        close = underlying.data["close"].copy()
        low = underlying.data["low"].copy()
        volume = underlying.data["volume"].copy()
        ret = close.pct_change()
        return ((-1 * ts_delta(ts_min(low, 5), 5)) * cs_rank(((ts_sum(ret, 240) - ts_sum(ret, 20)) / 220))) * ts_rank(
            volume, 5)

    @staticmethod
    def alpha_053(underlying):
        # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        close = underlying.data["close"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        inner = (close - low).replace(0, 0.0001)
        return -1 * ts_delta((((close - low) - (high - close)) / inner), 9)

    @staticmethod
    def alpha_054(underlying):
        # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        close = underlying.data["close"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        opn = underlying.data["open"].copy()
        inner = (low - high).replace(0, -0.0001)
        return -1 * (low - close) * (opn ** 5) / (inner * (close ** 5))

    @staticmethod
    def alpha_055(underlying):
        # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
        close = underlying.data["close"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        volume = underlying.data["volume"].copy()
        divisor = (ts_max(high, 12) - ts_min(low, 12)).replace(0, 0.0001)
        inner = (close - ts_min(low, 12)) / divisor
        df = ts_corr(cs_rank(inner), cs_rank(volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    # 本Alpha使用了cap|市值，暂未取到该值
    #    def alpha056(self):
    #        return (0 - (1 * (rank((sma(self.returns, 10) / sma(sma(self.returns, 2), 3))) * rank((self.returns * self.cap)))))

    @staticmethod
    def alpha_057(underlying):
        # Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        close = underlying.data["close"].copy()
        vwap = underlying.data["vwap"].copy()
        return 0 - (1 * ((close - vwap) / ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2)))

    # Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))

    # Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))

    @staticmethod
    def alpha_060(underlying):
        # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
        high = underlying.data["high"].copy()
        low = underlying.data["low"].copy()
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        divisor = (high - low).replace(0, 0.0001)
        inner = ((close - low) - (high - close)) * volume / divisor
        return - ((2 * scale(cs_rank(inner), 1)) - scale(cs_rank(ts_argmax(close, 10)), 1))

    @staticmethod
    def alpha_061(underlying):
        # Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        adv180 = ts_mean(volume, 180)
        return (cs_rank((vwap - ts_min(vwap, 16))) < cs_rank(ts_corr(vwap, adv180, 18))).astype('int')

    @staticmethod
    def alpha_062(underlying):
        # Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        high = underlying.data["high"].copy()
        low = underlying.data["low"].copy()
        opn = underlying.data["open"].copy()
        adv20 = ts_mean(volume, 20)
        return (cs_rank(ts_corr(vwap, ts_sum(adv20, 22), 10)) <
                cs_rank(((cs_rank(opn) + cs_rank(opn)) < (cs_rank(((high + low) / 2)) + cs_rank(high))))) * -1

    # Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)

    @staticmethod
    def alpha_064(underlying):
        # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        high = underlying.data["high"].copy()
        low = underlying.data["low"].copy()
        opn = underlying.data["open"].copy()
        adv120 = ts_mean(volume, 120)
        return ((cs_rank(ts_corr(ts_mean(((opn * 0.178404) + (low * (1 - 0.178404))), 13), ts_mean(adv120, 13), 17)) <
                 cs_rank(ts_delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1)

    @staticmethod
    def alpha_065(underlying):
        # Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        opn = underlying.data["open"].copy()
        adv60 = ts_mean(volume, 60)
        return (cs_rank(ts_corr(((opn * 0.00817205) + (vwap * (1 - 0.00817205))), ts_mean(adv60, 9), 6)) < cs_rank(
            (opn - ts_min(opn, 14)))) * -1

    @staticmethod
    def alpha_066(underlying):
        # Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
        vwap = underlying.data["vwap"].copy()
        high = underlying.data["high"].copy()
        low = underlying.data["low"].copy()
        opn = underlying.data["open"].copy()
        return ((cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear(((((low * 0.96633) + (
                low * (1 - 0.96633))) - vwap) / (opn - ((high + low) / 2))), 11), 7)) * -1)

    # # Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)

    @staticmethod
    def alpha_068(underlying):
        # Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
        volume = underlying.data["volume"].copy()
        high = underlying.data["high"].copy()
        low = underlying.data["low"].copy()
        close = underlying.data["close"].copy()
        adv15 = ts_mean(volume, 15)
        # 后者乘14，使比较双方处于同一水平
        return (ts_rank(ts_corr(cs_rank(high), cs_rank(adv15), 9), 14) < cs_rank(
            ts_delta(((close * 0.518371) + (low * (1 - 0.518371))), 2)) * 14) * -1

    # Alpha#69	 ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)

    # Alpha#70	 ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)

    @staticmethod
    def alpha_071(underlying):
        # Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        low = underlying.data["low"].copy()
        close = underlying.data["close"].copy()
        opn = underlying.data["open"].copy()
        adv180 = ts_mean(volume, 180)
        p1 = ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16)
        p2 = ts_rank(ts_decay_linear((cs_rank(((low + opn) - (vwap + vwap))).pow(2)), 16), 4)
        return max(p1, p2)

    @staticmethod
    def alpha_072(underlying):
        # Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        adv40 = ts_mean(volume, 40)
        return cs_rank(ts_decay_linear(ts_corr(((high + low) / 2), adv40, 9), 10)) / cs_rank(
            ts_decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))

    @staticmethod
    def alpha_073(underlying):
        # Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
        vwap = underlying.data["vwap"].copy()
        low = underlying.data["low"].copy()
        opn = underlying.data["open"].copy()
        p1 = cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3))
        p2 = ts_rank(ts_decay_linear(((ts_delta(((opn * 0.147155) + (low * (1 - 0.147155))), 2) / (
                (opn * 0.147155) + (low * (1 - 0.147155)))) * -1), 3), 17)
        return -1 * max(p1, p2)

    @staticmethod
    def alpha_074(underlying):
        # Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
        vwap = underlying.data["vwap"].copy()
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        high = underlying.data["high"].copy()
        adv30 = ts_mean(volume, 30)
        return (cs_rank(ts_corr(close, ts_mean(adv30, 37), 15)) < cs_rank(
            ts_corr(cs_rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), cs_rank(volume), 11))) * -1

    @staticmethod
    def alpha_075(underlying):
        # Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        low = underlying.data["low"].copy()
        adv50 = ts_mean(volume, 50)
        return (cs_rank(ts_corr(vwap, volume, 4)) < cs_rank(ts_corr(cs_rank(low), cs_rank(adv50), 12))).astype('int')

    # Alpha#76	 (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)

    @staticmethod
    def alpha_077(underlying):
        # Alpha#77	 min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        adv40 = ts_mean(volume, 40)
        p1 = cs_rank(ts_decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20))
        p2 = cs_rank(ts_decay_linear(ts_corr(((high + low) / 2), adv40, 3), 6))
        return min(p1, p2)

    # Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    @staticmethod
    def alpha_078(underlying):
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        low = underlying.data["low"].copy()
        adv40 = ts_mean(volume, 40)
        return cs_rank(ts_corr(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7)).pow(
            cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 6)))

    # Alpha#79	 (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))

    # Alpha#80	 ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)

    # # Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    @staticmethod
    def alpha_081(underlying):
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        adv10 = ts_mean(volume, 10)
        return ((cs_rank(log(ts_product(cs_rank(cs_rank(ts_corr(vwap, ts_sum(adv10, 50), 8)).pow(4)), 15))) < cs_rank(
            ts_corr(cs_rank(vwap), cs_rank(volume), 5))) * -1)

    #
    # # Alpha#82	 (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    #
    # # Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
    @staticmethod
    def alpha_083(underlying):
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        vwap = underlying.data["vwap"].copy()
        return ((cs_rank(ts_delay(((high - low) / (ts_sum(close, 5) / 5)), 2)) * cs_rank(
            cs_rank(volume))) / (
                        ((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close)))

    #
    # # Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
    @staticmethod
    def alpha_084(underlying):
        close = underlying.data["close"].copy()
        vwap = underlying.data["vwap"].copy()
        inner = ts_rank((vwap - ts_max(vwap, 15)), 21)
        return abs(inner).pow(ts_delta(close, 5)) * sign(inner)

    #
    # # Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
    @staticmethod
    def alpha_085(underlying):
        volume = underlying.data["volume"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        close = underlying.data["close"].copy()
        adv30 = ts_mean(volume, 30)
        return (cs_rank(ts_corr(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 10)).pow(
            cs_rank(ts_corr(ts_rank(((high + low) / 2), 4), ts_rank(volume, 10), 7))))

    #
    # # Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
    @staticmethod
    def alpha_086(underlying):
        open = underlying.data["open"].copy()
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        vwap = underlying.data["vwap"].copy()
        adv20 = ts_mean(volume, 20)
        # 后者乘以20，使比较双方处于同一水平
        return ((ts_rank(ts_corr(close, ts_mean(adv20, 15), 6), 20) < cs_rank(
            ((open + close) - (vwap + open))) * 20) * -1)

    #
    # # Alpha#87	 (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    #
    # # Alpha#88	 min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
    @staticmethod
    def alpha_088(underlying):
        open = underlying.data["open"].copy()
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        adv60 = ts_mean(volume, 60)
        p1 = cs_rank(ts_decay_linear(((cs_rank(open) + cs_rank(low)) - (cs_rank(high) + cs_rank(close))), 8))
        p2 = ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
        return min(p1, p2)

    #
    # # Alpha#89	 (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
    #
    # # Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
    #
    # # Alpha#91	 ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    #
    # # Alpha#92	 min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    @staticmethod
    def alpha_092(underlying):
        opn = underlying.data["open"].copy()
        volume = underlying.data["volume"].copy()
        close = underlying.data["close"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        adv30 = ts_mean(volume, 30)
        p1 = ts_rank(ts_decay_linear(((((high + low) / 2) + close) < (low + opn)), 15), 19)
        p2 = ts_rank(ts_decay_linear(ts_corr(cs_rank(low), cs_rank(adv30), 8), 7), 7)
        return min(p1, p2)

    #
    # # Alpha#93	 (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
    #
    # # Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    @staticmethod
    def alpha_094(underlying):
        vwap = underlying.data["vwap"].copy()
        volume = underlying.data["volume"].copy()
        adv60 = ts_mean(volume, 60)
        return ((cs_rank((vwap - ts_min(vwap, 12))).pow(
            ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)) * -1))

    # Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    @staticmethod
    def alpha_095(underlying):
        opn = underlying.data["open"].copy()
        volume = underlying.data["volume"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        adv40 = ts_mean(volume, 40)
        # 前者乘以12，使比较双方处于同一水平
        return (cs_rank((opn - ts_min(opn, 12))) * 12 < ts_rank(
            (cs_rank(ts_corr(ts_mean(((high + low) / 2), 19), ts_mean(adv40, 19), 13)).pow(5)), 12)).astype('int')

    #
    # # Alpha#96	 (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    @staticmethod
    def alpha_096(underlying):
        vwap = underlying.data["vwap"].copy()
        close = underlying.data["close"].copy()
        volume = underlying.data["volume"].copy()
        adv60 = ts_mean(volume, 60)
        p1 = ts_rank(ts_decay_linear(ts_corr(cs_rank(vwap), cs_rank(volume), 4), 4), 8)
        p2 = ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank(adv60, 4), 4), 13), 14), 13)
        return -1 * max(p1, p2)

    #
    # # Alpha#97	 ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    #
    # # Alpha#98	 (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    @staticmethod
    def alpha_098(underlying):
        vwap = underlying.data["vwap"].copy()
        volume = underlying.data["volume"].copy()
        opn = underlying.data["open"].copy()
        adv5 = ts_mean(volume, 5)
        adv15 = ts_mean(volume, 15)
        return (cs_rank(ts_decay_linear(ts_corr(vwap, ts_mean(adv5, 26), 5), 7)) - cs_rank(
            ts_decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(opn), cs_rank(adv15), 21), 9), 7), 8)))

    #
    # # Alpha#99	 ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
    @staticmethod
    def alpha_099(underlying):
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        volume = underlying.data["volume"].copy()
        adv60 = ts_mean(volume, 60)
        return ((cs_rank(ts_corr(ts_sum(((high + low) / 2), 20), ts_sum(adv60, 20), 9)) < cs_rank(
            ts_corr(low, volume, 6))) * -1)

    #
    # # Alpha#100	 (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))
    #
    # # Alpha#101	 ((close - open) / ((high - low) + .001))
    @staticmethod
    def alpha_101(underlying):
        close = underlying.data["close"].copy()
        opn = underlying.data["open"].copy()
        low = underlying.data["low"].copy()
        high = underlying.data["high"].copy()
        return (close - opn) / ((high - low) + 0.001)


if __name__ == "__main__":
    from alpha.data.underlying import *
    u = Underlying()
    alphas_101 = Alphas101("alphas_101")
    alpha_names = ["alpha_003"]
    alphas_101.cal_alphas(u, alpha_names=alpha_names, n_jobs=1)
    alphas_101.tst_alphas(alpha_names=alpha_names)
