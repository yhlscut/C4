from abc import ABC

import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

from auxiliary.utils import *

# --- START OF SQUEEZENET --

""" 
This is the standard SqueezeNet implementation included in PyTorch at:
https://github.com/pytorch/vision/blob/072d8b2280569a2d13b91d3ed51546d201a57366/torchvision/models/squeezenet.py
"""

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module, ABC):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))],
                         1)


class SqueezeNet(nn.Module, ABC):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()

        self.num_classes = num_classes

        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}: 1.0 or 1.1 expected".format(version=version))

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_0(pretrained: bool = False, **kwargs):
    """SqueezeNet model architecture from the paper 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
     <0.5MB model size': https://arxiv.org/abs/1602.07360.
    @param pretrained: if True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained: bool = False, **kwargs):
    """SqueezeNet 1.1 model from the official SqueezeNet repo:
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    Has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.
    @param pretrained: if True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


# --- END OF SQUEEZENET --


class CreateNet(nn.Module, ABC):
    def __init__(self, model):
        super(CreateNet, self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
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


class CreateNet_3stage(nn.Module, ABC):
    def __init__(self, num_model=2):
        super(CreateNet_3stage, self).__init__()
        self.submodel1 = CreateNet(squeezenet1_1(pretrained=True))
        self.submodel2 = CreateNet(squeezenet1_1(pretrained=True))
        self.submodel3 = CreateNet(squeezenet1_1(pretrained=True))

    def forward(self, x: torch.Tensor):
        """x has shape [bs, 3, h, w]"""

        output1 = self.submodel1(x)
        pred1 = torch.sum(torch.sum(output1, 2), 2)

        pred1 = torch.nn.functional.normalize(pred1, dim=1)
        correct_img1 = correct_image_nolinear(x, pred1)
        output2 = self.submodel2(correct_img1)
        pred2 = torch.sum(torch.sum(output2, 2), 2)
        pred2 = torch.nn.functional.normalize(pred2, dim=1)
        correct_img2 = correct_image_nolinear(x, torch.mul(pred1, pred2))
        output3 = self.submodel3(correct_img2)
        pred3 = torch.sum(torch.sum(output3, 2), 2)
        pred3 = torch.nn.functional.normalize(pred3, dim=1)
        return pred1, pred2, pred3

# if __name__ == '__main__':
#     network = CreateNet_1().to("cuda:0")
#     input = torch.randn([16, 3, 256, 256]).to("cuda:0")
#     label = torch.randn([16, 3]).to("cuda:0")
#     pred = network(input)
