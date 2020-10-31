import os
import time
from datetime import datetime

import torch

from auxiliary.utils import get_device
from classes.c4.models.Model import Model
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.training.Evaluator import Evaluator
from classes.training.MetricsTracker import MetricsTracker
from classes.training.VisdomHandler import VisdomHandler


class Trainer:

    def __init__(self, model: Model, opt, log_dir: str, log_name: str, visdom_port: int, single_stage: bool):
        self.__device = get_device()
        self.__evaluator = Evaluator()
        self.__single_stage = single_stage

        vh = VisdomHandler(port=visdom_port, env=opt.env + '-' + datetime.now().isoformat())
        self.__mt = MetricsTracker(vh, single_stage)

        self.__epochs = opt.nepoch
        self.__fold_num = opt.foldnum
        batch_size = opt.batch_size
        workers = opt.workers
        self.__log_dir = log_dir
        self.__log_name = log_name

        self.__model = model

        self.__training_loader = self.__create_data_loader(batch_size=batch_size, workers=workers, train=True)
        self.__test_loader = self.__create_data_loader(batch_size=1, workers=workers, train=False)

    def __create_data_loader(self, batch_size: int, workers: int, train: bool = True):
        dataset = ColorCheckerDataset(train=train, folds_num=self.__fold_num)
        print("{} set size ... : {}".format("Training" if train else "Test", len(dataset)))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    def train(self):

        print("\n Training starts... \n")

        for epoch in range(self.__epochs):

            self.__mt.reset_losses()

            # --- Training ---

            self.__model.train_mode()
            start = time.time()

            for _, data in enumerate(self.__training_loader):
                self.__model.reset_gradient()
                img, label, file_name = data
                img, label = img.to(self.__device), label.to(self.__device)
                train_loss = self.__model.compute_loss(img, label)
                self.__mt.update_train_loss(train_loss, epoch)
                self.__model.optimize()

            train_time = time.time() - start

            # --- Validation ---

            start = time.time()

            if epoch % 5 == 0:
                self.__evaluator.reset_errors()
                self.__model.evaluation_mode()

                with torch.no_grad():
                    for _, data in enumerate(self.__test_loader):
                        img, label, file_name = data
                        img, label = img.to(self.__device), label.to(self.__device)
                        val_loss = self.__model.compute_loss(img, label)
                        self.__mt.update_val_loss(val_loss, epoch)
                        self.__evaluator.add_error(val_loss if self.__single_stage else val_loss[3])

            val_time = time.time() - start

            train_loss, val_loss = self.__mt.get_train_loss_value(), self.__mt.get_val_loss_value()
            if self.__single_stage:
                print("Epoch: {},  Train_loss: {:.5},  Val_loss: {:.5}, T_time: {}, V_time: {}"
                      .format(epoch + 1, train_loss, val_loss, train_time, val_time))
            else:
                train_l3 = self.__mt.get_train_loss_tracker().get_step_loss(3)
                val_l3 = self.__mt.get_val_loss_tracker().get_step_loss(3)
                print("Epoch: {},  TL: {:.5}, TL3: {:.5},  VL: {:.5}, VL3: {:.5}, T_time: {:.5}, V_time: {:.5}"
                      .format(epoch + 1, train_loss, train_l3, val_loss, val_l3, train_time, val_time))

            metrics = self.__evaluator.compute_metrics()
            if 0 < self.__mt.get_val_loss_value() < self.__mt.get_best_val_loss():
                self.__mt.update_metrics(metrics)
                self.__model.save(os.path.join(self.__log_dir, "fold" + str(self.__fold_num) + ".pth"))

            self.__mt.log_metrics(self.__log_name, epoch)
