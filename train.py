import argparse
from train.trainer import Trainer
from train.settings import TrainSettings


def main():
    parser = argparse.ArgumentParser(description='LoFTR knowledge distillation.')
    parser.add_argument('--path', type=str, default='/home/kirill/development/datasets/BlendedMVS',
                        help='Path to the dataset.')
    parser.add_argument('--checkpoint_path', type=str, default='/home/kirill/development/models/LoFTR',
                        help='Path to the dataset.')

    opt = parser.parse_args()
    print(opt)

    settings = TrainSettings()
    trainer = Trainer(settings, opt.path, opt.checkpoint_path)
    trainer.train('LoFTR')


if __name__ == '__main__':
    main()