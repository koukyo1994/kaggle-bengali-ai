import numpy as np
import pandas as pd

from sklearn.metrics import recall_score


def macro_average_recall(prediction: np.ndarray, df: pd.DataFrame):
    grapheme = recall_score(
        df["grapheme_root"].values, prediction[:, 0], average="macro")
    vowel = recall_score(
        df["vowel_diacritic"].values, prediction[:, 1], average="macro")
    consonant = recall_score(
        df["consonant_diacritic"].values, prediction[:, 2], average="macro")
    return np.average([grapheme, vowel, consonant], weights=[2, 1, 1])
