from typing import NamedTuple

import numpy as np
import pandas as pd


class DTO(NamedTuple):
    train_X: pd.DataFrame
    train_y: np.ndarray
    test_X: pd.DataFrame
    
    oof_df: pd.DataFrame
    pred_df: pd.DataFrame

    def fetch_inputs(self):
        return self.train_X, self.train_y, self.test_X
    
    def fetch_outputs(self):
        return self.oof_df, self.pred_df