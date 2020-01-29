import numpy as np
import pandas as pd

from sklearn.metrics import recall_score


def macro_average_recall(prediction: np.ndarray, df: pd.DataFrame):
    grapheme = recall_score(
        prediction[:, 0], df["grapheme_root"].values, average="macro")
    vowel = recall_score(
        prediction[:, 1], df["vowel_diacritic"].values, average="macro")
    consonant = recall_score(
        prediction[:, 2], df["consonant_diacritic"].values, average="macro")
    return np.average([grapheme, vowel, consonant], weights=[2, 1, 1])
