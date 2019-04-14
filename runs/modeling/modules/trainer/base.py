from abc import ABCMeta, abstractmethod
from logging import getLogger, DEBUG

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


logger = getLogger(__name__)


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, n_split, kfold_seed, model_seed):
        self._n_split = n_split
        self._kf = self._fetch_kfold_methods(n_split, kfold_seed)
        self._params = self._set_model_params(model_seed)
    
    @property
    def n_split(self):
        return self._n_split
    
    @property
    def kf(self):
        return self._kf
    
    @property
    def params(self):
        return self._params
    
    @abstractmethod
    def _set_model_params(self, model_seed):
        raise NotImplementedError

    @abstractmethod
    def _train(self, dev_X, dev_y, val_X, val_y, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _predict(self, model, X):
        raise NotImplementedError 
    
    def cv(self, train_X, train_y, test_X, verbose=1, **kwargs):
        logger.debug('[START] start cv!')
        logger.debug('[PARAM] params: {}'.format(self.params))

        oof = np.zeros(len(train_X))
        predictions = np.zeros(len(test_X))
        scores = {'train': [], 'valid': []}
        feature_cols = list(train_X.columns)
        feature_importance_df = pd.DataFrame(columns=['fold', 'feature', 'importance'])

        for fold, (dev_idx, val_idx) in enumerate(self.kf.split(train_X, train_y)):
            logger.debug('[CV] fold {}/{}'.format(fold+1, self.n_split))

            dev_X, val_X = train_X.loc[dev_idx, :], train_X.loc[val_idx, :]
            dev_y, val_y = train_y[dev_idx], train_y[val_idx]

            model, dev_score, val_score, feature_importance = self._train(
                dev_X, dev_y, val_X, val_y, **kwargs
            )

            oof[val_idx] = self._predict(model, val_X)
            predictions += self._predict(model, test_X) / self.n_split

            scores['train'].append(dev_score)
            scores['valid'].append(val_score)

            if feature_importance is not None:
                partial_df = pd.DataFrame(columns=['fold', 'feature', 'importance'])
                partial_df['fold'] = fold + 1
                partial_df['feature'] = feature_cols
                partial_df['importance'] = feature_importance
                feature_importance_df = pd.concat([feature_importance_df, partial_df], axis=0)
        
        cv_score = self.calc_score(train_y, oof)

        if verbose >= 1:
            logger.info('[CV RESULTS]')
            logger.info('Num features: {}'.format(len(feature_cols)))
            logger.info('Num folds: {}'.format(self.n_split))
            logger.info('Train Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
                np.mean(scores['train']), 
                np.max(scores['train']), 
                np.min(scores['train']), 
                np.std(scores['train'])
            ))
            logger.info('Valid Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
                np.mean(scores['valid']), 
                np.max(scores['valid']), 
                np.min(scores['valid']), 
                np.std(scores['valid'])
            ))
            logger.info('CV Score: {:.5f}'.format(cv_score))

        return oof, predictions, feature_importance_df 
    
    def calc_score(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)
    
    def _fetch_kfold_methods(self, n_split, kfold_seed):
        return StratifiedKFold(n_splits=n_split, random_state=kfold_seed, shuffle=True)
