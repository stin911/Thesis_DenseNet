from __future__ import print_function, division
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
import Model as MTC

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", default=100, help="Set number of epochs", type=int)
    parser.add_argument("batch_size", default=1, help="Set the batch size", type=int)
    parser.add_argument("--learning_rate", default=1e-1, help="Set the learning rate", type=float)
    parser.add_argument("train_set", help="Path to dataset on local disk", type=Path)
    parser.add_argument("val_set", help="Path to dataset on local disk", type=Path)
    parser.add_argument("save_path", help="Path where to save weights", type=Path)
    parser.add_argument("--is_3d", action='store_true', default=False)

    args = parser.parse_args()
    test = MTC.BMSModel(args.epochs, args.batch_size, args.learning_rate, args.train_set, args.val_set,
                        args.save_path, args.is_3d)
    print(args.save_path)
    test.start_train(False)
    # test.infer()
