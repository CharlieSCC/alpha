import pandas as pd
import os
from alpha.config.config import *


class Underlying:
    def __init__(self):
        self.data = dict()
        self.load_data()

    def load_data(self):
        for key in DATA_KEYS:
            self.data[key] = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/1day_data/pv_ffill.h5"), key=key)