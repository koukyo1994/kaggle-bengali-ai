import torch

from catalyst.utils import get_device
from easydict import EasyDict as edict

from pathlib import Path
from typing import Union

from src.models import get_model


def load_model(config: edict, bin_path: Union[str, Path]):
    # config.model.pretrained = None
    model = get_model(config)
    state_dict = torch.load(bin_path, map_location=get_device())
    if "model_state_dict" in state_dict.keys():
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    return model
