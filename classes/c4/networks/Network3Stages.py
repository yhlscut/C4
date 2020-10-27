from abc import ABC

import torch

from auxiliary.utils import get_device
from classes.c4.networks.BaseNetwork import BaseNetwork
from classes.c4.networks.Network1Stage import Network1Stage


class Network3Stages(BaseNetwork, ABC):

    def __init__(self):
        super(Network3Stages, self).__init__()
        self.submodel1, self.submodel2, self.submodel3 = Network1Stage(), Network1Stage(), Network1Stage()

    @staticmethod
    def __correct_image_nonlinear(img: torch.Tensor, illuminant: torch.Tensor) -> torch.Tensor:
        # Correct the image
        nonlinear_illuminant = torch.pow(illuminant, 1.0 / 2.2)
        correction = nonlinear_illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(get_device())
        corrected_img = torch.div(img, correction + 1e-10)

        # Normalize the image
        max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
        max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return torch.div(corrected_img, max_img)

    def forward(self, x: torch.Tensor) -> tuple:
        """ x has shape [bs, 3, h, w] """

        pred1 = self.submodel1(x)
        correct_img1 = self.__correct_image_nonlinear(x, pred1)

        pred2 = self.submodel2(correct_img1)
        correct_img2 = self.__correct_image_nonlinear(x, torch.mul(pred1, pred2))

        pred3 = self.submodel3(correct_img2)

        return pred1, pred2, pred3
