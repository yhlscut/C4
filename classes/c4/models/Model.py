import math
from typing import Union

import torch
from torch.nn.functional import normalize


class Model:

    def __init__(self, device: torch.device):
        self._device = device
        self._network = None
        self.__optimizer = None

    def predict(self, image: torch.Tensor) -> Union[torch.Tensor, tuple]:
        pass

    def print_network(self):
        print(self._network)

    def log_network(self, log_name: str):
        open(log_name, 'a').write(str(self._network) + '\n')

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_file: str):
        torch.save(self._network.state_dict(), path_to_file)

    def load(self, path_to_pretrained: str):
        self._network.load_state_dict(torch.load(path_to_pretrained, map_location=self._device), strict=False)

    def set_optimizer(self, learning_rate: float):
        self.__optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)

    def reset_gradient(self):
        self.__optimizer.zero_grad()

    def optimize(self):
        self.__optimizer.step()

    @staticmethod
    def get_loss(prediction: torch.Tensor, label: torch.Tensor, safe_v: float = 0.999999) -> torch.Tensor:
        """ Angular loss """
        dot = torch.sum(normalize(prediction, dim=1) * normalize(label, dim=1), dim=1)
        dot = torch.clamp(dot, -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle)
