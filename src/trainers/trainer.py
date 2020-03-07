import numpy as np
import torch
import torch.nn as nn

from fastprogress import progress_bar

from .state import State


class BatchCallback:
    def on_loader_start(self, epoch: int, num_epochs: int):
        raise NotImplementedError

    def on_batch_start(self, batch):
        raise NotImplementedError

    def on_batch_end(self, batch):
        raise NotImplementedError


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixupOrCutmixCallback(BatchCallback):
    def __init__(self,
                 alpha=1.0,
                 mixup_prob=0.5,
                 cutmix_prob=0.5,
                 no_aug_epochs=0):
        assert alpha >= 0, "alpha must be>=0"
        assert 1 >= mixup_prob >= 0, "mixup_prob must be within 1 and 0"
        assert 1 >= cutmix_prob >= 0, "cutmix_prob must be within 1 and 0"
        assert 1 >= mixup_prob + cutmix_prob, \
            "sum of mixup_prob and cutmix_prob must be lower than 1"
        self.alpha = alpha
        self.lam = 1
        self.is_needed = True
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.no_action_prob = 1 - (mixup_prob + cutmix_prob)
        self.no_aug_epochs = no_aug_epochs

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

    def on_loader_start(self, epoch: int, num_epochs: int):
        if num_epochs - epoch <= self.no_aug_epochs:
            self.is_needed = False

    def on_batch_start(self, batch):
        if not self.is_needed:
            return batch

        dice = np.random.choice([0, 1, 2],
                                p=(self.mixup_prob, self.cutmix_prob,
                                   self.no_action_prob))
        self.dice = dice
        if dice == 0:
            if self.alpha > 0:
                self.lam = np.random.beta(self.alpha, self.alpha)
            else:
                self.lam = 1
            self.index = torch.randperm(batch["images"].shape[0])
            self.index.to(self.device)

            batch["images"] = self.lam * batch["images"] + \
                (1 - self.lam) * batch["images"][self.index]
        elif dice == 1:
            self.index = torch.randperm(batch["images"].shape[0])
            self.index.to(self.device)

            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1
                bbx1, bby1, bbx2, bby2 = rand_bbox(batch["images"].size(), lam)
                batch["images"][:, :, bbx1:bbx2, bby1:bby2] = \
                    batch["images"][self.index, :, bbx1:bbx2, bby1:bby2]
                self.lam = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1) /
                    (batch["images"].size()[-1] * batch["images"].size()[-2]))
        else:
            pass
        return batch


def train_one_epoch(model, criterion, optimizer, scheduler, loader,
                    current_epoch, batch_callbacks):
    model.train()
    avg_loss = 0.0

    for loader_output in progress_bar(loader, leave=False):
        images = loader_output["images"].to(device)
        targets = loader_output["targets"].to(device)
        preds = model(images)
