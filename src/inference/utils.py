import torch

from catalyst.utils import get_device
from easydict import EasyDict as edict

from pathlib import Path
from typing import Union


def load_model(model_class, config: edict, bin_path: Union[str, Path]):
    model_params = config.model
    model_params.pretrained = None
    model = model_class(**model_params)
    model.load_state_dict(torch.load(bin_path, map_location=get_device()))
    return model
