from classes.training.VisdomHandler import VisdomHandler


class LossTracker(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_angular_loss(self) -> float:
        return self.avg

    def update_visualization(self, vh: VisdomHandler, epoch: int, name: str):
        vh.update(epoch, loss=self.avg, name=name)
