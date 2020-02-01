from easydict import EasyDict as edict

from .metrics import TotalAverageRecall, MacroAverageRecall, AverageRecall
from .save import SaveWeightsCallback
from .augmentation import MixupOrCutmixCallback


def get_callbacks(config: edict):
    callbacks = []
    for callback in config.callbacks:
        name = list(callback.keys())[0]
        params = callback[name]
        if globals().get(name) is not None:
            if params is not None:
                callbacks.append(globals().get(name)(**params))  # type: ignore
            else:
                callbacks.append(globals().get(name)())  # type: ignore
    return callbacks
