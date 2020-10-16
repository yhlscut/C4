from abc import ABC

import torch.utils.model_zoo as model_zoo
from torch import nn

from classes.c4.squeezenet.SqueezeNet import SqueezeNet


class Model(nn.Module, ABC):
    __model_urls = {
        'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
        'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    }

    def _squeezenet1_0(self, pretrained: bool = False, **kwargs):
        """SqueezeNet model architecture from the paper 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
         <0.5MB model size': https://arxiv.org/abs/1602.07360.
        @param pretrained: if True, returns a model pre-trained on ImageNet
        """
        model = SqueezeNet(version=1.0, **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(self.__model_urls['squeezenet1_0']))
        return model

    def _squeezenet1_1(self, pretrained: bool = False, **kwargs):
        """SqueezeNet 1.1 model from the official SqueezeNet repo:
        <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
        Has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.
        @param pretrained: if True, returns a model pre-trained on ImageNet
        """
        model = SqueezeNet(version=1.1, **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(self.__model_urls['squeezenet1_1']))
        return model
