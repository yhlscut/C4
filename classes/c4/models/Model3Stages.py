from abc import ABC

import torch

from auxiliary.utils import get_device
from classes.c4.models.Model import Model
from classes.c4.models.Model1Stage import Model1Stage


class Model3Stages(Model, ABC):

    def __init__(self):
        super(Model3Stages, self).__init__()
        self.submodel1 = Model1Stage()
        self.submodel2 = Model1Stage()
        self.submodel3 = Model1Stage()

    @staticmethod
    def __correct_image_nolinear(img, ill):
        # nolinear img, linear ill , return non-linear img
        nonlinear_ill = torch.pow(ill, 1.0 / 2.2)
        correction = nonlinear_ill.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(get_device())
        corrected_img = torch.div(img, correction + 1e-10)
        img_max = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
        img_max = img_max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        img_normalize = torch.div(corrected_img, img_max)
        return img_normalize

    def forward(self, x: torch.Tensor):
        """x has shape [bs, 3, h, w]"""

        output1 = self.submodel1(x)
        pred1 = torch.sum(torch.sum(output1, 2), 2)
        pred1 = torch.nn.functional.normalize(pred1, dim=1)
        correct_img1 = self.__correct_image_nolinear(x, pred1)

        output2 = self.submodel2(correct_img1)
        pred2 = torch.sum(torch.sum(output2, 2), 2)
        pred2 = torch.nn.functional.normalize(pred2, dim=1)
        correct_img2 = self.__correct_image_nolinear(x, torch.mul(pred1, pred2))

        output3 = self.submodel3(correct_img2)
        pred3 = torch.sum(torch.sum(output3, 2), 2)
        pred3 = torch.nn.functional.normalize(pred3, dim=1)

        return pred1, pred2, pred3
