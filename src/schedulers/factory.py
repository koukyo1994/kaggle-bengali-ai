from typing import Optional, Union

from easydict import EasyDict as edict
from torch.optim.lr_scheduler import (ReduceLROnPlateau, CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts)

Scheduler = Optional[
    Union[ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts]]


def get_scheduler(optimizer, config: edict) -> Scheduler:
    params = config.scheduler.params
    name = config.scheduler.name
    scheduler: Scheduler = None
    if name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, **params)
    elif name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, **params)
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmRestarts(optimizer, **params)

    return scheduler
