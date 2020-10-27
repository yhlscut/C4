import math
from typing import Union

import torch
from torch.nn.functional import normalize


class Model:

    def __init__(self, device: torch.device):
        self._device = device
        self._network = None

    def predict(self, image: torch.Tensor) -> Union[torch.Tensor, tuple]:
        pass

    @staticmethod
    def get_loss(prediction: torch.Tensor, label: torch.Tensor, safe_v: float = 0.999999) -> torch.Tensor:
        """ Angular loss """
        dot = torch.sum(normalize(prediction, dim=1) * normalize(label, dim=1), dim=1)
        dot = torch.clamp(dot, -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def load_pretrained(self, path_to_pretrained: str):
        self._network.load_state_dict(torch.load(path_to_pretrained, map_location=self._device), strict=False)
