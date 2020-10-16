from __future__ import print_function

import argparse
import datetime
import json
import os
import time

import torch.optim as optim
import torch.utils.data
import visdom

from auxiliary.dataset import *
from auxiliary.utils import *


def main(log_name):

    # Set device
    device = get_device()

    # Visualization
    vis = visdom.Visdom(port=8097, env=opt.env + '-' + save_path)
    win_curve = vis.line(X=np.array([0]), Y=np.array([0]))

    # Load data
    dataset_train = ColorChecker(train=True, folds_num=opt.foldnum)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.workers)
    print('len_dataset_train:', len(dataset_train))

    dataset_test = ColorChecker(train=False, folds_num=opt.foldnum)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=opt.workers)
    print('len_dataset_test:', len(dataset_test))
    print('training fold %d' % opt.foldnum)

    # Create the network
    network = Model3Stages().to(device)
    if opt.pth_path != '':
        print('loading pretrained model')
        network.submodel1.load_state_dict(torch.load(opt.pth_path))
        network.submodel2.load_state_dict(torch.load(opt.pth_path))
        network.submodel3.load_state_dict(torch.load(opt.pth_path))
    with open(log_name, 'a') as f:
        f.write(str(network) + '\n')

    # Set the optimizer
    lrate = opt.lrate
    optimizer = optim.Adam(network.parameters(), lr=lrate)

    # --- TRAINING ---

    train_loss, train_loss1, train_loss2, train_loss3 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    val_loss, val_loss1, val_loss2, val_loss3 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    best_val_loss = 100.0
    best_mean, best_median, best_trimean = 100.0, 100.0, 100.0
    best_bst25, best_wst25, best_pct95 = 100.0, 100.0, 100.0

    print('Start training.....')
    for epoch in range(opt.nepoch):

        # Optimize
        train_loss.reset()
        train_loss1.reset()
        train_loss2.reset()
        train_loss3.reset()

        network.train()

        start = time.time()
        for i, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            img, label, fn = data
            img, label = img.to(device), label.to(device)
            pred1, pred2, pred3 = network(img)

            loss1 = get_angular_loss(pred1, label)
            loss2 = get_angular_loss(torch.mul(pred1, pred2), label)
            loss3 = get_angular_loss(torch.mul(torch.mul(pred1, pred2), pred3), label)
            loss = opt.alpha1 * loss1 + opt.alpha2 * loss2 + (1.0 - opt.alpha1 - opt.alpha2) * loss3
            loss.backward()

            train_loss.update(loss.item())
            train_loss1.update(loss1.item())
            train_loss2.update(loss2.item())
            train_loss3.update(loss3.item())

            optimizer.step()

        time_use1 = time.time() - start
        try:
            vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([train_loss.avg]),
                win=win_curve,
                name='train_loss'
            )
            vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([train_loss1.avg]),
                win=win_curve,
                name='train_loss1'
            )
            vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([train_loss2.avg]),
                win=win_curve,
                name='train_loss2'
            )
            vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([train_loss3.avg]),
                win=win_curve,
                name='train_loss3'
            )
        except:
            print('visdom error......')

        # Evaluate
        time_use2 = 0
        val_loss.reset()
        val_loss1.reset()
        val_loss2.reset()
        val_loss3.reset()
        with torch.no_grad():
            if epoch % 5 == 0:
                network.eval()
                start = time.time()
                errors = []
                for _, data in enumerate(dataloader_test):
                    img, label, fn = data
                    img, label = img.to(device), label.to(device)
                    pred1, pred2, pred3 = network(img)

                    loss1 = get_angular_loss(pred1, label)
                    loss2 = get_angular_loss(torch.mul(pred1, pred2), label)
                    loss3 = get_angular_loss(torch.mul(torch.mul(pred1, pred2), pred3), label)
                    loss = opt.alpha1 * loss1 + opt.alpha2 * loss2 + (1.0 - opt.alpha1 - opt.alpha2) * loss3

                    val_loss.update(loss.item())
                    val_loss1.update(loss1.item())
                    val_loss2.update(loss2.item())
                    val_loss3.update(loss3.item())

                    errors.append(loss3.item())

                time_use2 = time.time() - start
                try:
                    vis.updateTrace(
                        X=np.array([epoch]),
                        Y=np.array([val_loss.avg]),
                        win=win_curve,
                        name='val loss'
                    )
                    vis.updateTrace(
                        X=np.array([epoch]),
                        Y=np.array([val_loss1.avg]),
                        win=win_curve,
                        name='val_loss1'
                    )
                    vis.updateTrace(
                        X=np.array([epoch]),
                        Y=np.array([val_loss2.avg]),
                        win=win_curve,
                        name='val_loss2'
                    )
                    vis.updateTrace(
                        X=np.array([epoch]),
                        Y=np.array([val_loss3.avg]),
                        win=win_curve,
                        name='val_loss3'
                    )
                except:
                    print('visdom error......')

        mean, median, trimean, bst25, wst25, pct95 = evaluate(errors)
        print('Epoch: %d, Train_loss: %f, Train_loss3: %f, Val_loss: %f, Val_loss3: %f, T_Time: %f, V_time: %f'
              % (epoch, train_loss.avg, train_loss3.avg, val_loss.avg, val_loss3.avg, time_use1, time_use2))

        if 0 < val_loss3.avg < best_val_loss:
            best_val_loss = val_loss3.avg
            best_mean, best_median, best_trimean = mean, median, trimean
            best_bst25, best_wst25, best_pct95 = bst25, wst25, pct95
            torch.save(network.state_dict(), '%s/fold%d.pth' % (dir_name, opt.foldnum))

        log_table = {
            "train_loss": train_loss.avg,
            "val_loss": val_loss.avg,
            "epoch": epoch,
            "lr": lrate,
            "best_val_loss": best_val_loss,
            "mean": best_mean,
            "median": best_median,
            "trimean": best_trimean,
            "bst25": best_bst25,
            "wst25": best_wst25,
            "pct95": best_pct95,
            "alpha1": opt.alpha1,
            "alpha2": opt.alpha2,
        }
        with open(log_name, 'a') as f:
            f.write('json_stats: ' + json.dumps(log_table) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=20)
    parser.add_argument('--lrate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--env', type=str, default='main', help='visdom environment')
    parser.add_argument('--pth_path', type=str, default='')
    parser.add_argument('--alpha1', default=0.1, type=float, help='alpha1')
    parser.add_argument('--alpha2', default=0.2, type=float, help='alpha2')
    parser.add_argument('--foldnum', type=int, default=0, help='fold number')
    opt = parser.parse_args()
    print(opt)

    now = datetime.datetime.now()
    save_path = now.isoformat()
    dir_name = './log/C4_sq_3stage'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    log_name = os.path.join(dir_name, 'log_fold' + str(opt.foldnum) + '.txt')

    main(log_name)
