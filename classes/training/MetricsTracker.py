import json
from typing import Union

from classes.training.LossTracker import LossTracker
from classes.training.MultipleLossTracker import MultipleLossTracker
from classes.training.VisdomHandler import VisdomHandler


class MetricsTracker:

    def __init__(self, vh: VisdomHandler, single_stage: bool):
        self.__vh = vh
        self.__train_loss = LossTracker() if single_stage else MultipleLossTracker()
        self.__val_loss = LossTracker() if single_stage else MultipleLossTracker()
        self.__best_val_loss = 100.0
        self.__best_mean, self.__best_median, self.__best_trimean = 100.0, 100.0, 100.0
        self.__best_bst25, self.__best_wst25, self.__best_pct95 = 100.0, 100.0, 100.0

    def update_train_loss(self, loss: Union[float, list], epoch: int):
        self.__train_loss.update(loss)
        self.__train_loss.update_visualization(self.__vh, epoch, name="Training loss")

    def update_val_loss(self, loss: Union[float, list], epoch: int):
        self.__val_loss.update(loss)
        self.__train_loss.update_visualization(self.__vh, epoch, name="Validation loss")

    def get_train_loss_value(self) -> float:
        return self.__train_loss.get_angular_loss()

    def get_val_loss_value(self) -> float:
        return self.__val_loss.get_angular_loss()

    def get_train_loss_tracker(self) -> Union[LossTracker, MultipleLossTracker]:
        return self.__train_loss

    def get_val_loss_tracker(self) -> Union[LossTracker, MultipleLossTracker]:
        return self.__val_loss

    def get_best_val_loss(self) -> float:
        return self.__best_val_loss

    def reset_losses(self):
        self.__train_loss.reset()
        self.__val_loss.reset()

    def update_metrics(self, metrics: dict):
        self.__best_val_loss = self.__val_loss.get_angular_loss()
        self.__best_mean = metrics["mean"]
        self.__best_median = metrics["median"]
        self.__best_trimean = metrics["trimean"]
        self.__best_bst25 = metrics["bst25"]
        self.__best_wst25 = metrics["wst25"],
        self.__best_pct95 = metrics["pct95"]

    def log_metrics(self, log_name: str, epoch: int):
        log_table = {
            "train_loss": self.__train_loss.get_angular_loss(),
            "val_loss": self.__val_loss.get_angular_loss(),
            "best_val_loss": self.__best_val_loss,
            "mean": self.__best_mean,
            "median": self.__best_median,
            "trimean": self.__best_trimean,
            "bst25": self.__best_bst25,
            "wst25": self.__best_wst25,
            "pct95": self.__best_pct95
        }
        open(log_name, 'a').write("Stats for epoch {}:\n {} \n".format(epoch, json.dumps(log_table, indent=2)))
