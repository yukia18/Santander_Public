import argparse
from pathlib import Path
from logging import getLogger, basicConfig, DEBUG

from modules import DataTransfer, TrainerAgency


formatter = '%(message)s'
basicConfig(level=DEBUG, format=formatter)
logger = getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()

    dir_input_path = f'../../results/inputs/{args.dir}'
    dir_output_path = f'../../results/outputs/{args.dir}'

    transfer = DataTransfer(dir_input_path, dir_output_path)
    dto = transfer.load()

    agency = TrainerAgency()
    trainer = agency.load_trainer(args.model, n_split=2)
    oof, predictions, feature_importance_df = trainer.cv(
        dto.train_X, dto.train_y, dto.test_X, verbose=1
    )

    prefix = f'{args.dir}_{args.model}_'
    oof_df, pred_df = dto.fetch_outputs()
    transfer.save(oof_df, pred_df, oof, predictions, feature_importance_df, prefix=prefix)


if __name__ == '__main__':
    main()