import json

from classes.data.AverageMeter import AverageMeter


class MetricsTracker:

    def __init__(self):
        self.__train_loss, self.__val_loss = AverageMeter(), AverageMeter()
        self.__best_val_loss = 100.0
        self.__best_mean, self.__best_median, self.__best_trimean = 100.0, 100.0, 100.0
        self.__best_bst25, self.__best_wst25, self.__best_pct95 = 100.0, 100.0, 100.0

    def update_train_loss(self, loss: float):
        self.__train_loss.update(loss)

    def update_val_loss(self, loss: float):
        self.__val_loss.update(loss)

    def get_train_loss(self) -> float:
        return self.__train_loss.avg

    def get_val_loss(self) -> float:
        return self.__val_loss.avg

    def get_best_val_loss(self) -> float:
        return self.__best_val_loss

    def reset_losses(self):
        self.__train_loss.reset()
        self.__val_loss.reset()

    def update_metrics(self, metrics: dict):
        self.__best_val_loss = self.__val_loss.avg
        self.__best_mean = metrics["mean"]
        self.__best_median = metrics["median"]
        self.__best_trimean = metrics["trimean"]
        self.__best_bst25 = metrics["bst25"]
        self.__best_wst25 = metrics["wst25"],
        self.__best_pct95 = metrics["pct95"]

    def log_metrics(self, log_name: str, epoch: int):
        log_table = {
            "train_loss": self.__train_loss.avg,
            "val_loss": self.__val_loss.avg,
            "best_val_loss": self.__best_val_loss,
            "mean": self.__best_mean,
            "median": self.__best_median,
            "trimean": self.__best_trimean,
            "bst25": self.__best_bst25,
            "wst25": self.__best_wst25,
            "pct95": self.__best_pct95
        }
        open(log_name, 'a').write("Stats for epoch {}:\n {} \n".format(epoch, json.dumps(log_table, indent=2)))
