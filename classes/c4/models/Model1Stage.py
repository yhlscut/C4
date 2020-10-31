import torch
from torch.autograd import Variable

from classes.c4.models.Model import Model
from classes.c4.networks.Network1Stage import Network1Stage


class Model1Stage(Model):

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._network = Network1Stage().to(self._device)

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        return self._network(img)

    def compute_loss(self, img: torch.Tensor, label: torch.Tensor) -> float:
        pred = self.predict(img)
        loss = Variable(self.get_angular_loss(pred, label), requires_grad=True)
        loss.backward()
        return loss.item()
