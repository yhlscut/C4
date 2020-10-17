import math
import re

import numpy as np
import torch


def get_device() -> torch.device:
    """
    Returns the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    :return: the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    """
    device_type = "cpu"

    if device_type == "cpu":
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", device_type):
        if not torch.cuda.is_available():
            print("WARNING: running on cpu since device {} is not available".format(device_type))
            return torch.device("cpu")
        return torch.device(device_type)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(device_type))


class AverageMeter(object):
    # Computes and stores the average and current value

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_angular_loss(vec1, vec2):
    safe_v = 0.999999
    illum_normalized1 = torch.nn.functional.normalize(vec1, dim=1)
    illum_normalized2 = torch.nn.functional.normalize(vec2, dim=1)
    dot = torch.sum(illum_normalized1 * illum_normalized2, dim=1)
    dot = torch.clamp(dot, -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    loss = torch.mean(angle)
    return loss


def correct_image_nolinear(img, ill):
    # nolinear img, linear ill , return non-linear img
    nonlinear_ill = torch.pow(ill, 1.0 / 2.2)
    correct = nonlinear_ill.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(get_device())
    correc_img = torch.div(img, correct + 1e-10)
    img_max = torch.max(torch.max(torch.max(correc_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    img_max = img_max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    img_normalize = torch.div(correc_img, img_max)
    return img_normalize


def evaluate(errors):
    errors = sorted(errors)

    def g(f):
        return np.percentile(errors, f * 100)

    median = g(0.5)
    mean = np.mean(errors)
    trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
    bst25 = np.mean(errors[:int(0.25 * len(errors))])
    wst25 = np.mean(errors[int(0.75 * len(errors)):])
    pct95 = g(0.95)
    return mean, median, trimean, bst25, wst25, pct95
