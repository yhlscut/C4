from typing import Union

import torch

from classes.c4.models.Model import Model
from classes.c4.networks.Network3Stages import Network3Stages


class Model3Stages(Model):

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._network = Network3Stages().to(self._device)

    def predict(self, img: torch.Tensor, return_raw_predictions: bool = False) -> Union[torch.Tensor, tuple]:
        pred1, pred2, pred3 = self._network(img)
        return (pred1, pred2, pred3) if return_raw_predictions else torch.mul(torch.mul(pred1, pred2), pred3)
