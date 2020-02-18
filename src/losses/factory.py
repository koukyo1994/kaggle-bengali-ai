import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict


class BengaliCrossEntropyLoss(nn.Module):
    def __init__(self,
                 n_grapheme: int,
                 n_vowel: int,
                 n_consonant: int,
                 weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.cross_entropy = nn.CrossEntropyLoss()
        self.weights = weights

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

        return self.weights[0] * self.cross_entropy(
            grapheme_pred, grapheme_true) + \
            self.weights[1] * self.cross_entropy(vowel_pred, vowel_true) + \
            self.weights[2] * self.cross_entropy(
                consonant_pred, consonant_true)


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=1,
                 gamma=2,
                 logits=False,
                 reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(
                inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)


class MultiHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.size_average = size_average

    def forward(self, output, y):
        output_y = output[torch.arange(0,
                                       y.size()[0]).long().cuda(),
                          y.data.cuda()].view(-1, 1)
        loss = output - output_y + self.margin
        loss[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()] = 0
        loss[loss < 0] = 0
        if (self.p != 1):
            loss = torch.pow(loss, self.p)

        if (self.weight is not None):
            loss = loss * self.weight

        loss = torch.sum(loss)
        if (self.size_average):
            loss /= output.size()[0]
        return loss


class BengaliFocalLoss(nn.Module):
    def __init__(self,
                 n_grapheme: int,
                 n_vowel: int,
                 n_consonant: int,
                 weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.focal = FocalLoss()
        self.weights = weights

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

        return self.weights[0] * self.focal(
            grapheme_pred, grapheme_true) + \
            self.weights[1] * self.focal(vowel_pred, vowel_true) + \
            self.weights[2] * self.focal(
                consonant_pred, consonant_true)


class BengaliMultiMarginLoss(nn.Module):
    def __init__(self,
                 n_grapheme: int,
                 n_vowel: int,
                 n_consonant: int,
                 weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.margin = MultiHingeLoss()
        self.weights = weights

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

        return self.weights[0] * self.margin(
            grapheme_pred, grapheme_true) + \
            self.weights[1] * self.margin(vowel_pred, vowel_true) + \
            self.weights[2] * self.margin(
                consonant_pred, consonant_true)


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


class OHEMLoss(nn.Module):
    def __init__(self, rate=0.7):
        super().__init__()
        self.rate = rate

    def forward(self, pred, target):
        batch_size = pred.size(0)
        ohem_cls_loss = F.cross_entropy(
            pred, target, reduction="none", ignore_index=-1)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size(0), int(batch_size * self.rate))
        if keep_num < sorted_ohem_loss.size(0):
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss


class BengaliOHEMLoss(nn.Module):
    def __init__(self,
                 n_grapheme: int,
                 n_vowel: int,
                 n_consonant: int,
                 weights=(1.0, 1.0, 1.0),
                 rate=0.7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.ohem = OHEMLoss(rate=rate)
        self.weights = weights

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

        return self.weights[0] * self.ohem(
            grapheme_pred, grapheme_true) + \
            self.weights[1] * self.ohem(vowel_pred, vowel_true) + \
            self.weights[2] * self.ohem(
                consonant_pred, consonant_true)


class GraphemeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, true):
        return self.loss(pred, true[:, 0])


def get_loss(config: edict):
    name = config.loss.name
    params = config.loss.params
    if name == "bce":
        criterion = BengaliBCELoss(**params)
    elif name == "cross_entropy":
        criterion = BengaliCrossEntropyLoss(**params)  # type: ignore
    elif name == "grapheme":
        criterion = GraphemeLoss()  # type: ignore
    elif name == "margin":
        criterion = BengaliMultiMarginLoss(**params)  # type: ignore
    elif name == "focal":
        criterion = BengaliFocalLoss(**params)  # type: ignore
    elif name == "ohem":
        criterion = BengaliOHEMLoss(**params)  # type: ignore
    else:
        raise NotImplementedError
    return criterion
