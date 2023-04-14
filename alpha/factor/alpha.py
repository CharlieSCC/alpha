import os
import alphalens
from tqdm import tqdm
from joblib import Parallel, delayed
from alpha.config.config import DATA_PATH
from alpha.factor.operators import *


class Alphas:

    def __init__(self, name):
        self.name = name

    def lst_alphas(self):
        return sorted(list(filter(lambda m: m.startswith("alpha"), dir(self))))

    def cal_alphas(self, underlying, n_jobs: int, alpha_names=None,):
        if not os.path.exists(os.path.join(DATA_PATH, "Ashare_data/factor_data/")):
            os.makedirs(os.path.join(DATA_PATH, "Ashare_data/factor_data/"))
        if alpha_names is None:
            alpha_names = self.lst_alphas()
        else:
            alpha_names = alpha_names

        def _cal_alphas(uly, m):
            res = getattr(self, m)(uly)
            res.to_hdf(os.path.join(DATA_PATH, "Ashare_data/factor_data/{}_{}.h5".format(self.name, m)), key=m)

        # 在线程池中计算所有alpha
        if n_jobs > 1:
            Parallel(n_jobs)(delayed(_cal_alphas)(underlying, m) for m in alpha_names)
        else:
            for m in tqdm(alpha_names):
                _cal_alphas(underlying, m)

    def tst_alphas(self, quantiles=5, alpha_names=None, measure="open"):
        if alpha_names is None:
            alpha_names = self.lst_alphas()
        else:
            alpha_names = alpha_names
        for m in alpha_names:
            print("======================================{} profiling======================================".format(m))
            fac = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/factor_data/{}_{}.h5".format(self.name, m)), key=m).iloc[1215:]
            fac.index = pd.to_datetime(fac.index)
            close = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/1day_data/pv.h5"), key=measure).iloc[1215:]
            close.index = pd.to_datetime(close.index)
            factor_data = alphalens.utils.get_clean_factor_and_forward_returns(fac.stack(),
                                                                               close.shift(-1),
                                                                               quantiles=quantiles,
                                                                               max_loss=1.0)
            alphalens.tears.create_full_tear_sheet(factor_data)



