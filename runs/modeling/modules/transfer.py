from pathlib import Path
from logging import getLogger, DEBUG

import pandas as pd

from modules.dto import DTO


logger = getLogger(__name__)


class DataTransfer:
    def __init__(self, dir_input_path, dir_output_path, key='ID_code', target='target'):
        self._dir_input_path = Path(dir_input_path)
        self._dir_output_path = Path(dir_output_path)
        self._key = key
        self._target = target
    
    @property
    def dir_input_path(self):
        return self._dir_input_path
    
    @property
    def dir_output_path(self):
        return self._dir_output_path
    
    @property
    def key(self):
        return self._key
    
    @property
    def target(self):
        return self._target
    
    def load(self):
        train_df = pd.read_pickle(self.dir_input_path.joinpath('train_X.pkl'))
        test_df = pd.read_pickle(self.dir_input_path.joinpath('test_X.pkl'))

        train_X = train_df.loc[:, [col not in [self.key, self.target] for col in train_df.columns]]
        train_y = train_df[self.target].values
        test_X = test_df.loc[:, [col != self.key for col in test_df.columns]]

        oof_df = pd.DataFrame({self.key: train_df[self.key].values})
        pred_df = pd.DataFrame({self.key: test_df[self.key].values})

        logger.debug('[SHAPE] train_X.shape: {}, test_X.shape: {}'.format(
            train_X.shape, test_X.shape
        ))

        return DTO(train_X, train_y, test_X, oof_df, pred_df)
    
    def save(self, oof_df, pred_df, oof, predictions, feature_importance_df, prefix=''):
        self.dir_output_path.mkdir(parents=True, exist_ok=True)

        oof_df[self.target] = oof
        oof_df.to_csv(
            self.dir_output_path.joinpath(f'{prefix}oof.csv'),
            index=False
        )

        pred_df[self.target] = predictions
        pred_df.to_csv(
            self.dir_output_path.joinpath(f'{prefix}predictions.csv'),
            index=False
        )

        feature_importance_df.to_csv(
            self.dir_output_path.joinpath(f'{prefix}feature_importance.csv'),
            index=False
        )

        logger.info('saved results to {}'.format(str(self.dir_output_path)))
