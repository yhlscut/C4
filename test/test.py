from __future__ import print_function

import argparse
import os

import torch.utils.data
from torch.autograd import Variable

from auxiliary.utils import get_device
from classes.c4.models.Model1Stage import Model1Stage
from classes.c4.models.Model3Stages import Model3Stages
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.training.Evaluator import Evaluator

"""
Reported results for C4-SqueezeNetFC4 on Color Checker:
Mean: 1.35 | Median: 0.88 | Tri-mean: 0.99 | Best 25%: 0.28 | Worst 25% 3.21
"""

STAGES = 1
WORKERS = 20
PRETRAINED_MODEL_FOLDERS = {1: "c4_sq_1stage", 3: "c4_sq_3stage"}
BASE_PATH_TO_PRETRAINED = os.path.join("trained_models", PRETRAINED_MODEL_FOLDERS[STAGES])
PTH_PATH_0 = os.path.join(BASE_PATH_TO_PRETRAINED, "fold0.pth")
PTH_PATH_1 = os.path.join(BASE_PATH_TO_PRETRAINED, "fold1.pth")
PTH_PATH_2 = os.path.join(BASE_PATH_TO_PRETRAINED, "fold2.pth")


def main():
    device = get_device()
    models = {1: Model1Stage, 3: Model3Stages}
    pth_paths = {0: opt.pth_path0, 1: opt.pth_path1, 2: opt.pth_path2}
    workers = opt.workers
    stages = opt.stages
    evaluator = Evaluator()

    model = models[stages](device)

    for i in range(3):
        test_set = ColorCheckerDataset(train=False, folds_num=i)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=workers)
        print('\n Length of fold {}: {} \n'.format(i, len(test_set)))

        model.load(path_to_pretrained=pth_paths[i])
        model.evaluation_mode()

        with torch.no_grad():
            for _, data in enumerate(dataloader):
                img, label, file_name = data
                img, label = Variable(img.to(device)), Variable(label.to(device))
                pred = model.predict(img)
                loss = model.get_angular_loss(pred, label)
                evaluator.add_error(loss.item())
                print('\t - Input: %s, AE: %f' % (file_name[0], loss.item()))

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["pct95"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=STAGES, help="number of stages of the model (i.e. either 1 or 3)")
    parser.add_argument("--workers", type=int, default=WORKERS, help="number of data loading workers")
    parser.add_argument("--pth_path0", type=str, default=PTH_PATH_0)
    parser.add_argument("--pth_path1", type=str, default=PTH_PATH_1)
    parser.add_argument("--pth_path2", type=str, default=PTH_PATH_2)
    opt = parser.parse_args()
    print("\n Configuration: {} \n".format(opt))
    main()
