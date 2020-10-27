from __future__ import print_function

import argparse
import datetime
import os
import time

import torch.utils.data

from auxiliary.utils import *
from classes.c4.models.Model1Stage import Model1Stage
from classes.data.ColorChecker import ColorCheckerDataset
from classes.data.Evaluator import Evaluator
from classes.data.MetricsTracker import MetricsTracker
from classes.data.VisdomHandler import VisdomHandler

VISDOM_PORT = 8097
VISDOM_ENV = "mian"
LOG_DIR = os.path.join("train", "log", "C4_sq_1stage")
BATCH_SIZE = 16
NUM_EPOCHS = 2000
WORKERS = 30
LEARNING_RATE = 0.0003
PTH_PATH = ""
FOLD_NUM = 0


def main():
    device = get_device()
    evaluator = Evaluator()
    mt = MetricsTracker()

    learning_rate = opt.lrate
    epochs = opt.nepoch
    fold_num = opt.foldnum
    batch_size = opt.batch_size
    workers = opt.workers

    os.makedirs(LOG_DIR, exist_ok=True)
    log_name = os.path.join(LOG_DIR, 'log_fold_' + str(fold_num) + '.txt')

    vh = VisdomHandler(port=VISDOM_PORT, env=opt.env + '-' + datetime.datetime.now().isoformat())

    # Training set
    training_set = ColorCheckerDataset(train=True, folds_num=fold_num)
    training_loader = torch.utils.data.DataLoader(training_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=workers)

    # Validation set
    test_set = ColorCheckerDataset(train=False, folds_num=fold_num)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=opt.workers)

    print('Training set size: ...', len(training_set))
    print('Test set size: .......', len(test_set))

    print('\n * Training fold %d * \n' % fold_num)

    # Model
    model = Model1Stage(device)

    if opt.pth_path != "":
        print('\n Loading pretrained model... \n')
        model.load(opt.pth_path)

    model.print_network()
    model.log_network(log_name)
    model.set_optimizer(learning_rate)

    print("\n Training starts... \n")

    for epoch in range(epochs):

        mt.reset_losses()

        # --- Training ---

        model.train_mode()
        start = time.time()

        for _, data in enumerate(training_loader):
            model.reset_gradient()
            img, label, file_name = data
            img, label = img.to(device), label.to(device)
            pred = model.predict(img)
            loss = model.get_loss(pred, label)
            loss.backward()
            mt.update_train_loss(loss.item())
            model.optimize()

        vh.update(epoch, loss=mt.get_train_loss(), name="train loss")

        training_time = time.time() - start

        # --- Validation ---

        start = time.time()

        if epoch % 5 == 0:
            evaluator.reset_errors()
            model.evaluation_mode()

            with torch.no_grad():
                for _, data in enumerate(test_loader):
                    img, label, file_name = data
                    img, label = img.to(device), label.to(device)
                    pred = model.predict(img)
                    loss = model.get_loss(pred, label)
                    mt.update_val_loss(loss.item())
                    evaluator.add_error(loss.item())

            vh.update(epoch, loss=mt.get_val_loss(), name="val loss")

        validation_time = time.time() - start

        print("Epoch: {},  Train_loss: {},  Val_loss: {}, T_Time: {}, V_time: {}"
              .format(epoch + 1, mt.get_train_loss(), mt.get_val_loss(), training_time, validation_time))

        metrics = evaluator.compute_metrics()
        if 0 < mt.get_val_loss() < mt.get_best_val_loss():
            mt.update_metrics(metrics)
            model.save(os.path.join(LOG_DIR, "fold" + str(fold_num) + ".pth"))

        mt.log_metrics(log_name, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
