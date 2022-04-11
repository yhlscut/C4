from classes.training.LossTracker import LossTracker
from classes.training.VisdomHandler import VisdomHandler


class MultipleLossTracker:

    def __init__(self):
        self.__loss = LossTracker()
        self.__loss1 = LossTracker()
        self.__loss2 = LossTracker()
        self.__loss3 = LossTracker()

    def reset(self):
        self.__loss.reset()
        self.__loss1.reset()
        self.__loss2.reset()
        self.__loss3.reset()

    def update(self, losses: list):
        self.__loss.update(losses[0])
        self.__loss1.update(losses[1])
        self.__loss2.update(losses[2])
        self.__loss3.update(losses[3])

    def update_visualization(self, vh: VisdomHandler, epoch: int, name: str):
        self.__loss.update_visualization(vh, epoch, name)
        self.__loss1.update_visualization(vh, epoch, name)
        self.__loss2.update_visualization(vh, epoch, name)
        self.__loss3.update_visualization(vh, epoch, name)

    def get_step_loss(self, step: int) -> float:
        if step == 1:
            return self.__loss1.get_angular_loss()
        if step == 2:
            return self.__loss2.get_angular_loss()
        if step == 3:
            return self.__loss3.get_angular_loss()
        raise ValueError("{} is not a valid step, must be either 1, 2 or 3!".format(step))

    def get_angular_loss(self):
        return self.__loss.get_angular_loss()
