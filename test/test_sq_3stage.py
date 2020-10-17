from __future__ import print_function

import argparse

import torch.utils.data
from auxiliary.dataset import *
from torch.autograd import Variable

from auxiliary.utils import *


def main(opt):
    val_loss = AverageMeter()
    errors = []
    device = get_device()

    # Create network
    network = Model3Stages().to(device)
    network.eval()

    for i in range(3):
        dataset_test = ColorCheckerDataset(train=False, folds_num=i)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=opt.workers)
        len_dataset_test = len(dataset_test)
        print("Len_fold:", len_dataset_test)
        if i == 0:
            pth_path = opt.pth_path0
        elif i == 1:
            pth_path = opt.pth_path1
        elif i == 2:
            pth_path = opt.pth_path2

        # Load parameters
        network.load_state_dict(torch.load(pth_path, map_location=device))
        for i, data in enumerate(dataloader_test):
            img, label, file_name = data
            img = Variable(img.to(device))
            label = Variable(label.to(device))
            pred1, pred2, pred3, = network(img)
            loss = get_angular_loss(torch.mul(torch.mul(pred1, pred2), pred3), label)
            val_loss.update(loss.item())
            errors.append(loss.item())
            print('Model: %s, AE: %f' % (file_name[0], loss.item()))

    mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
    print('Mean: %f, Med: %f, tri: %f, bst: %f, wst: %f, pct: %f' % (mean, median, trimean, bst25, wst25, pct95))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, help="number of data loading workers", default=20)
    parser.add_argument("--lrate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--pth_path0", type=str, default="trained_models/fold0.pth")
    parser.add_argument("--pth_path1", type=str, default="trained_models/fold1.pth")
    parser.add_argument("--pth_path2", type=str, default="trained_models/fold2.pth")
    parser.add_argument("--alpha1", default=0.33, type=float, help="alpha1")
    parser.add_argument("--alpha2", default=0.33, type=float, help="alpha2")
    opt = parser.parse_args()
    print(opt)
    main(opt)
