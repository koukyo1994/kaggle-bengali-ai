from typing import Union

from easydict import EasyDict as edict
from torch.optim import Adam, SGD

Optimizer = Union[Adam, SGD]


def get_optimizer(model, config: edict) -> Optimizer:
    name = config.optimizer.name
    params = config.optimizer.params
    if name == "Adam":
        optimizer = Adam(model.parameters(), **params)
    elif name == "SGD":
        optimizer = Adam(model.parameters(), **params)
    else:
        raise NotImplementedError
    return optimizer
