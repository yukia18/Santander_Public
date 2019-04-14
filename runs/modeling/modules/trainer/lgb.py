import lightgbm as lgb

from .base import BaseTrainer


class LGBTrainer(BaseTrainer):
    def _set_model_params(self, model_seed):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 3,
            'max_depth': -1,
            'learning_rate': 0.01,
            'min_data_in_leaf': 80, 
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'feature_fraction': 0.7, 
            'lambda_l1': 1.5402103791851314, 
            'lambda_l2': 3.5484117894847667,
            'bagging_seed': model_seed,
            'feature_fraction_seed': model_seed,
            'seed': model_seed,
            'save_binary': True,
            'device': 'cpu',
            'verbosity': -1,
        }

        return params

    def _train(self, dev_X, dev_y, val_X, val_y, **kwargs):
        lgdev = lgb.Dataset(dev_X, dev_y)
        lgval = lgb.Dataset(val_X, val_y)

        model = lgb.train(
            self.params, lgdev, 
            valid_sets=[lgdev, lgval],
            num_boost_round=100000,
            early_stopping_rounds=1000,
            verbose_eval=1000,
        )

        dev_score = model.best_score['training']['auc']
        val_score = model.best_score['valid_1']['auc']
        feature_importance = model.feature_importance()

        return model, dev_score, val_score, feature_importance

    def _predict(self, model, X):
        return model.predict(X, num_iteration=model.best_iteration)
