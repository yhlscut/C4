from __future__ import print_function

import argparse

import torch.utils.data
from torch.autograd import Variable

from auxiliary.utils import *
from classes.c4.networks.Network3Stages import Network3Stages
from classes.data.ColorChecker import ColorCheckerDataset


def main():
    device = get_device()
    val_loss = AverageMeter()
    pth_paths = {0: opt.pth_path0, 1: opt.pth_path1, 2: opt.pth_path2}
    workers = opt.workers
    errors = []

    network = Network3Stages().to(device)
    network.eval()

    for i in range(3):
        test_set = ColorCheckerDataset(train=False, folds_num=i)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=workers)
        print('\n Length of fold #{}: {} \n'.format(i, len(test_set)))

        # Load the pretrained model
        network.load_state_dict(torch.load(pth_paths[i], map_location=device), strict=False)
        network.eval()

        with torch.no_grad():
            for _, data in enumerate(dataloader):
                img, label, file_name = data
                img, label = Variable(img.to(device)), Variable(label.to(device))
                pred1, pred2, pred3, = network(img)
                loss = get_angular_loss(torch.mul(torch.mul(pred1, pred2), pred3), label)
                val_loss.update(loss.item())
                errors.append(loss.item())
                print('\t - Input: %s, AE: %f' % (file_name[0], loss.item()))

    mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
    print('\n Mean: %f, Med: %f, tri: %f, bst: %f, wst: %f, pct: %f' % (mean, median, trimean, bst25, wst25, pct95))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, help="number of data loading workers", default=20)
    parser.add_argument("--lrate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--pth_path0", type=str, default="trained_models/c4_sq_3stage/fold0.pth")
    parser.add_argument("--pth_path1", type=str, default="trained_models/c4_sq_3stage/fold1.pth")
    parser.add_argument("--pth_path2", type=str, default="trained_models/c4_sq_3stage/fold2.pth")
    parser.add_argument("--alpha1", default=0.33, type=float, help="alpha1")
    parser.add_argument("--alpha2", default=0.33, type=float, help="alpha2")
    opt = parser.parse_args()
    main()
