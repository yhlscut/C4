from typing import Union

import torch
from torch.autograd import Variable

from classes.c4.models.Model import Model
from classes.c4.networks.Network3Stages import Network3Stages


class Model3Stages(Model):

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._network = Network3Stages().to(self._device)

    def predict(self, img: torch.Tensor, return_raw_predictions: bool = False) -> Union[torch.Tensor, tuple]:
        pred1, pred2, pred3 = self._network(img)
        return (pred1, pred2, pred3) if return_raw_predictions else torch.mul(torch.mul(pred1, pred2), pred3)

    def load_submodules(self, path_to_pretrained: str):
        self._network.submodel1.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))
        self._network.submodel2.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))
        self._network.submodel3.load_state_dict(torch.load(path_to_pretrained, map_location=self._device))

    @staticmethod
    def get_multiply_accumulated_loss(l1: torch.Tensor,
                                      l2: torch.Tensor,
                                      l3: torch.Tensor,
                                      a1: float = 0.33,
                                      a2: float = 0.33) -> torch.Tensor:
        return a1 * l1 + a2 * l2 + (1.0 - a1 - a2) * l3

    def compute_loss(self, img: torch.Tensor, label: torch.Tensor) -> list:
        pred1, pred2, pred3 = self.predict(img, return_raw_predictions=True)
        l1 = Variable(self.get_angular_loss(pred1, label), requires_grad=True)
        l2 = Variable(self.get_angular_loss(torch.mul(pred1, pred2), label), requires_grad=True)
        l3 = Variable(self.get_angular_loss(torch.mul(torch.mul(pred1, pred2), pred3), label), requires_grad=True)
        loss = Variable(self.get_multiply_accumulated_loss(l1, l2, l3), requires_grad=True)
        loss.backward()
        return [loss.item(), l1.item(), l2.item(), l3.item()]
