import re

import torch

DEVICE_TYPE = "cpu"


def get_device() -> torch.device:
    """
    Returns the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    :return: the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    """

    if DEVICE_TYPE == "cpu":
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", DEVICE_TYPE):
        if not torch.cuda.is_available():
            print("WARNING: running on cpu since device {} is not available".format(DEVICE_TYPE))
            return torch.device("cpu")
        return torch.device(DEVICE_TYPE)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(DEVICE_TYPE))
