from .trainer import LGBTrainer
from .trainer import CBTrainer


class TrainerAgency:    
    def load_trainer(self, model, n_split=5, kfold_seed=42, model_seed=42):
        if model == 'lgb':
            trainer = LGBTrainer(n_split, kfold_seed, model_seed)

        elif model == 'cb':
            trainer = CBTrainer(n_split, kfold_seed, model_seed)

        else:
            raise NotImplementedError(f'{model} trainer not exists.')
        
        return trainer