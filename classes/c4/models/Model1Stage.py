from abc import ABC

import torch
import torch.nn as nn

from classes.c4.models.Model import Model


class Model1Stage(Model, ABC):

    def __init__(self):
        super().__init__()
        self.squeezenet1_1 = nn.Sequential(*list(self._squeezenet1_1(pretrained=True).children())[0][:12])
        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        x = self.squeezenet1_1(x)
        x = self.fc(x)
        return x
