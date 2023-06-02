class BackTest:
    def __init__(self, capital, fee_rate, tax_rate):
        """

        Parameters
        ----------
        info
        ---------------------------------------------------------
        date     |      stock_id     |     y_pred    |     y
        ---------------------------------------------------------

        ---------------------------------------------------------
        """
        self.capital = capital
        self.fee_rate = fee_rate
        self.tax_rate = tax_rate

    @staticmethod
    def get_top_stock(alpha, pct):
        alpha = alpha.sort_values(by=["date", "y_pred"], ascending=[True, False]).reset_index(drop=True)
        alpha = alpha.groupby("date").apply(lambda x: x.iloc[:int(len(x)*pct)])
        return alpha.reset_index(drop=True).set_index(["date", "stock_id"])
    @staticmethod
    def get_bottom_stock(alpha, pct):
        alpha = alpha.sort_values(by=["date", "y_pred"], ascending=[True, False]).reset_index(drop=True)
        alpha = alpha.groupby("date").apply(lambda x: x.iloc[-int(len(x)*pct):])
        return alpha.reset_index(drop=True).set_index(["date", "stock_id"])
    @staticmethod
    def get_position_and_turnover(alpha):
        alpha["position"] = alpha.groupby("date")["y_pred"].apply(lambda x: x - x + 1/len(x)/2)
        alpha["turnover"] = alpha["position"].unstack().sort_index().fillna(0).diff().stack()
        return alpha.fillna(0)

    def get_daily_pnl(self, alpha, pct):
        alpha_long = self.get_top_stock(alpha, pct)
        alpha_long  = self.get_position_and_turnover(alpha_long)
        alpha_long["cost"] = alpha_long["turnover"].abs() * self.fee_rate + alpha_long["turnover"].apply(lambda x: min(x, 0)).abs() * self.tax_rate
        daily_ret_long = alpha_long["position"] * alpha_long["ret"] - alpha_long["cost"]

        alpha_bottom = self.get_bottom_stock(alpha, pct)
        alpha_bottom = self.get_position_and_turnover(alpha_bottom)
        alpha_bottom["cost"] = alpha_bottom["turnover"].abs() * self.fee_rate + alpha_bottom["turnover"].apply(lambda x: min(x, 0)).abs() * self.tax_rate
        daily_ret_short = alpha_bottom["position"] * alpha_bottom["ret"] - alpha_bottom["cost"]
        return daily_ret_long.groupby("date").sum().shift(1).fillna(0) - daily_ret_short.groupby("date").sum().shift(1).fillna(0)
