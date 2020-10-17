from __future__ import print_function

import argparse

import torch.utils.data
from torch.autograd import Variable

from auxiliary.utils import *


def main():
    val_loss = AverageMeter()
    errors = []

    # Create the network
    SqueezeNet = squeezenet1_1(pretrained=True)
    network = Model1Stage(SqueezeNet).to(device)
    network.eval()

    for i in range(3):
        ############################################test fold 0############################################
        dataset_test = ColorCheckerDataset(train=False, folds_num=i)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False,
                                                      num_workers=opt.workers)
        len_dataset_test = len(dataset_test)
        print('Len_fold:', len(dataset_test))
        if i == 0:
            pth_path = opt.pth_path0
        elif i == 1:
            pth_path = opt.pth_path1
        elif i == 2:
            pth_path = opt.pth_path2
            # load parameters
        network.load_state_dict(torch.load(pth_path))
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                img, label, file_name = data
                img = Variable(img.cuda())
                label = Variable(label.cuda())
                pred = network(img)
                pred_ill = torch.nn.functional.normalize(torch.sum(torch.sum(pred, 2), 2), dim=1)
                loss = get_angular_loss(pred_ill, label)
                val_loss.update(loss.item())
                errors.append(loss.item())
                print('Model: %s, AE: %f' % (file_name[0], loss.item()))

    mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
    print('Mean: %f, Med: %f, tri: %f, bst: %f, wst: %f, pct: %f' % (mean, median, trimean, bst25, wst25, pct95))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=20)
    parser.add_argument('--lrate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--pth_path0', type=str)
    parser.add_argument('--pth_path1', type=str)
    parser.add_argument('--pth_path2', type=str)
    opt = parser.parse_args()

    main()
