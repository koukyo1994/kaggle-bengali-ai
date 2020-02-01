from typing import List

import numpy as np

import torch

from catalyst.dl.callbacks import CriterionCallback
from catalyst.dl.core import RunnerState


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixupOrCutmixCallback(CriterionCallback):
    def __init__(self,
                 fields: List[str] = [
                     "images",
                 ],
                 alpha=1.0,
                 on_train_only=True,
                 mixup_prob=0.5,
                 cutmix_prob=0.5,
                 **kwargs):
        assert len(fields) > 0, \
            "At least one field is required"
        assert alpha >= 0, "alpha must be>=0"
        assert 1 >= mixup_prob >= 0, "mixup_prob must be within 1 and 0"
        assert 1 >= cutmix_prob >= 0, "cutmix_prob must be within 1 and 0"
        assert 1 >= mixup_prob + cutmix_prob, \
            "sum of mixup_prob and cutmix_prob must be lower than 1"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.no_action_prob = 1 - (mixup_prob + cutmix_prob)

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        dice = np.random.choice(
            [0, 1, 2],
            p=[self.mixup_prob, self.cutmix_prob, self.no_action_prob])
        self.dice = dice
        if dice == 0:
            if self.alpha > 0:
                self.lam = np.random.beta(self.alpha, self.alpha)
            else:
                self.lam = 1
            self.index = torch.randperm(state.input[self.fields[0]].shape[0])
            self.index.to(state.device)

            for f in self.fields:
                state.input[f] = self.lam * state.input[f] + \
                    (1 - self.lam) * state.input[f][self.index]
        elif dice == 1:
            self.index = torch.randperm(state.input[self.fields[0]].shape[0])
            self.index.to(state.device)

            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1
                bbx1, bby1, bbx2, bby2 = rand_bbox(
                    state.input[self.fields[0]].size(), lam)
                for f in self.fields:
                    state.input[f][:, :, bbx1:bbx2, bby1:bby2] = \
                        state.input[f][self.index, :, bbx1:bbx2, bby1:bby2]
                self.lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                                (state.input[self.fields[0]].size()[-1] *
                                 state.input[self.fields[0]].size()[-2]))
        else:
            pass

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        if self.dice == 0 or self.dice == 1:
            pred = state.output[self.output_key]
            y_a = state.input[self.input_key]
            y_b = state.input[self.input_key][self.index]
            loss = self.lam * criterion(pred, y_a) + \
                (1 - self.lam) * criterion(pred, y_b)
            return loss
        else:
            return super()._compute_loss(state, criterion)
