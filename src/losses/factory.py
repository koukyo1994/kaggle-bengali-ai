import torch.nn as nn

from easydict import EasyDict as edict


class BengaliCrossEntropyLoss(nn.Module):
    def __init__(self, n_grapheme: int, n_vowel: int, n_consonant: int):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, true):
        head = 0
        tail = self.n_grapheme
        grapheme_pred = pred[:, head:tail]
        grapheme_true = true[:, 0]

        head = tail
        tail = head + self.n_vowel
        vowel_pred = pred[:, head:tail]
        vowel_true = true[:, 1]

        head = tail
        tail = head + self.n_consonant
        consonant_pred = pred[:, head:tail]
        consonant_true = true[:, 2]

        return self.cross_entropy(grapheme_pred, grapheme_true) + \
            self.cross_entropy(vowel_pred, vowel_true) + \
            self.cross_entropy(consonant_pred, consonant_true)


class BengaliBCELoss(nn.Module):
    def __init__(self, n_grapheme: int, n_vowel: int, n_consonant: int):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, true):
        head = 0
        tail = self.n_grapheme
        grapheme_pred = pred[:, head:tail]
        grapheme_true = true[:, head:tail]

        head = tail
        tail = head + self.n_vowel
        vowel_pred = pred[:, head:tail]
        vowel_true = true[:, head:tail]

        head = tail
        tail = head + self.n_consonant
        consonant_pred = pred[:, head:tail]
        consonant_true = true[:, head:tail]

        return self.bce(grapheme_pred, grapheme_true) + \
            self.bce(vowel_pred, vowel_true) + \
            self.bce(consonant_pred, consonant_true)


def get_loss(config: edict):
    name = config.loss.name
    params = config.loss.params
    if name == "bce":
        criterion = BengaliBCELoss(**params)
    elif name == "cross_entropy":
        criterion = BengaliCrossEntropyLoss(**params)  # type: ignore
    else:
        raise NotImplementedError
    return criterion
