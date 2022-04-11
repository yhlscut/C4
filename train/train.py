from __future__ import print_function

import argparse
import os

from auxiliary.utils import get_device
from classes.c4.models.Model1Stage import Model1Stage
from classes.c4.models.Model3Stages import Model3Stages
from classes.training.Trainer import Trainer

STAGES = 1
VISDOM_PORT = 8097
VISDOM_ENV = "mian"
BATCH_SIZE = 2
NUM_EPOCHS = 2000
WORKERS = 20
LEARNING_RATE = 0.0003
FOLD_NUM = 0
MODELS_DIRS = {1: "C4_sq_1stage", 3: "C4_sq_3stage"}
LOG_DIR = os.path.join("log", MODELS_DIRS[STAGES])
PTH_PATH = os.path.join("trained_models", MODELS_DIRS[1], "fold" + str(FOLD_NUM) + ".pth")


def main():
    device = get_device()
    stages = opt.stages
    fold_num = opt.foldnum
    pth_path = opt.pth_path
    learning_rate = opt.lrate

    os.makedirs(LOG_DIR, exist_ok=True)
    log_name = os.path.join(LOG_DIR, 'log_fold_' + str(fold_num) + '.txt')

    print('\n * Training fold %d * \n' % fold_num)

    models = {1: Model1Stage, 3: Model3Stages}
    model = models[stages](device)

    if pth_path != "":
        if stages == 1:
            model.load(pth_path)
        elif stages == 3:
            model.load_submodules(pth_path)
        else:
            raise ValueError("{} is an invalid number of stages, could not load pretrained model!".format(stages))

    model.print_network()
    model.log_network(log_name)
    model.set_optimizer(learning_rate)

    trainer = Trainer(model, opt, LOG_DIR, log_name, VISDOM_PORT, single_stage=stages == 1)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=STAGES, help="number of stages of the model (i.e. either 1 or 3)")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=NUM_EPOCHS, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=WORKERS, help='number of data loading workers')
    parser.add_argument('--lrate', type=float, default=LEARNING_RATE, help='learning rate')
    parser.add_argument('--env', type=str, default=VISDOM_ENV, help='visdom environment')
    parser.add_argument('--pth_path', type=str, default=PTH_PATH)
    parser.add_argument('--foldnum', type=int, default=FOLD_NUM, help='fold number')
    opt = parser.parse_args()
    print("\n Configuration: {} \n".format(opt))
    main()
