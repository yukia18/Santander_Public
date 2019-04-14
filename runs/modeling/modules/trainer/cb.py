import catboost as cb
from torch.cuda import is_available

from .base import BaseTrainer


class CBTrainer(BaseTrainer):
    def _set_model_params(self, model_seed):
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': model_seed,
            'allow_writing_files': False,
            'task_type': 'GPU' if is_available() else 'CPU',
        }

        return params

    def _train(self, dev_X, dev_y, val_X, val_y, **kwargs):
        model = cb.CatBoostClassifier(**self.params)
        model.fit(dev_X, dev_y, eval_set=(val_X, val_y), use_best_model=True, verbose=200)

        dev_pred = model.predict_proba(dev_X)[:,1]
        dev_score = self.calc_score(dev_y, dev_pred)

        val_pred = model.predict_proba(val_X)[:,1]
        val_score = self.calc_score(val_y, val_pred)

        feature_importance = model.feature_importances_

        return model, dev_score, val_score, feature_importance

    def _predict(self, model, X):
        return model.predict_proba(X)[:,1]