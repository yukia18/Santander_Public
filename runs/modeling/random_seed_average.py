import argparse
from pathlib import Path
from logging import getLogger, basicConfig, DEBUG

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from modules import DataTransfer, TrainerAgency


formatter = '%(message)s'
basicConfig(level=DEBUG, format=formatter)
logger = getLogger(__name__)


def random_seed_average_cv(train_X, train_y, test_X, args, verbose=1):
    oof_array = np.zeros((args.num_random_seed_average, len(train_X)))
    predictions_array = np.zeros((args.num_random_seed_average, len(test_X)))
    feature_importance_df = pd.DataFrame(columns=['fold', 'feature', 'importance'])

    agency = TrainerAgency()
    random_seed_sampler = check_random_state(42)
    cv_scores = []

    for i in range(args.num_random_seed_average):
        logger.debug('[RUN] random seed average: {}/{}'.format(
            i+1, args.num_random_seed_average
        ))
        
        seed = random_seed_sampler.randint(0, 1000)

        trainer = agency.load_trainer(args.model, n_split=4, kfold_seed=seed, model_seed=seed)
        oof, predictions, df = trainer.cv(train_X, train_y, test_X, verbose=1)
        cv_scores.append(trainer.calc_score(train_y, oof))

        oof_array[i] = oof
        predictions_array[i] = predictions
        feature_importance_df = pd.concat([feature_importance_df, df], axis=0)
    
    oof = np.mean(oof_array, axis=0)
    predictions = np.mean(predictions_array, axis=0)
    mean_cv_score = trainer.calc_score(train_y, oof)
    
    if verbose >= 1:
        logger.info('{} CV Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
            args.num_random_seed_average, 
            np.mean(cv_scores), 
            np.max(cv_scores), 
            np.min(cv_scores), 
            np.std(cv_scores)
        ))
        logger.info('random {} seed average cv score: {:.5f}'.format(
            args.num_random_seed_average, mean_cv_score
        ))
    
    return oof, predictions, feature_importance_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-n_rsa', '--num_random_seed_average', type=int, default=5)
    args = parser.parse_args()

    dir_input_path = f'../../results/inputs/{args.dir}'
    dir_output_path = f'../../results/outputs/{args.dir}'

    transfer = DataTransfer(dir_input_path, dir_output_path)
    dto = transfer.load()

    train_X, train_y, test_X = dto.fetch_inputs()
    oof, predictions, feature_importance_df = random_seed_average_cv(
        train_X, train_y, test_X, args, verbose=1
    )

    prefix = f'{args.dir}_rsa_{args.model}_'
    oof_df, pred_df = dto.fetch_outputs()
    transfer.save(oof_df, pred_df, oof, predictions, feature_importance_df, prefix=prefix)


if __name__ == '__main__':
    main()