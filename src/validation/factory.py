import numpy as np
import pandas as pd

from typing import List, Tuple

from easydict import EasyDict as edict
from sklearn.model_selection import KFold, train_test_split


def no_fold(df: pd.DataFrame,
            config: edict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config.val.params
    idx = np.arange(len(df))
    trn_idx, val_idx = train_test_split(idx, **params)
    return [(trn_idx, val_idx)]


def kfold(df: pd.DataFrame,
          config: edict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config.val.params
    kf = KFold(shuffle=True, **params)
    splits = list(kf.split(df))
    return splits


def get_validation(df: pd.DataFrame,
                   config: edict) -> List[Tuple[np.ndarray, np.ndarray]]:
    name: str = config.val.name

    func = globals().get(name)
    if func is None:
        raise NotImplementedError

    return func(df, config)
