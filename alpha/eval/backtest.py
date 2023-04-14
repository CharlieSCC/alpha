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
        alpha = alpha.groupby("date").apply(lambda x: x.sort_values("y_pred", ascending=False).iloc[:int(len(x)*pct)])
        return alpha.reset_index().set_index(["date", "stock_id"])

    @staticmethod
    def get_position_and_turnover(alpha):
        alpha["position"] = alpha.groupby("date").apply(lambda x: 1/len(x))
        alpha["turnover"] = alpha["position"].unstack().sort_index().diff().stack()
        return alpha.fillna(0)

    def get_daily_pnl(self, alpha, pct):
        alpha = self.get_top_stock(alpha, pct)
        alpha = self.get_position_and_turnover(alpha)
        alpha["cost"] = alpha["turnover"].abs() * self.fee_rate + alpha["turnover"].apply(lambda x: min(x, 0)).abs() * self.tax_rate
        daily_ret = alpha["position"] * alpha["y"] - alpha["cost"]
        return daily_ret * self.capital





