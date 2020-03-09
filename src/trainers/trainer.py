import numpy as np
import torch

from fastprogress import progress_bar


class BatchCallback:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

    def on_loader_start(self, state: dict):
        return state

    def on_batch_start(self, state: dict):
        state["batch"]["images"] = state["batch"]["images"].to(self.device)
        state["batch"]["targets"] = state["batch"]["tragets"].to(self.device)
        return state

    def on_batch_end(self, state: dict):
        return state


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


class CriterionCallback(BatchCallback):
    def __init__(self, criterion):
        self.criterion = criterion
        super().__init__()

    def on_loader_start(self, state: dict):
        self.avg_loss = 0.0
        self.n_steps = len(state["loader"])
        return state

    def _calc_loss(self, state: dict):
        batch = state["batch"]
        target = batch["targets"]
        pred = state["pred"]
        loss = self.criterion(pred, target)
        return loss

    def on_batch_end(self, state: dict):
        optimizer = state["optimizer"]
        scheduler = state["scheduler"]
        optimizer.zero_grad()

        loss = self._calc_loss(state)
        loss.backward()
        optimizer.step()
        scheduler.step()
        self.avg_loss += loss.item() / self.n_steps
        state["avg_loss"] = self.avg_loss
        return state


class MixupOrCutmixCallback(CriterionCallback):
    def __init__(self,
                 criterion,
                 alpha=1.0,
                 mixup_prob=0.5,
                 cutmix_prob=0.5,
                 no_aug_epochs=0):
        super().__init__(criterion)

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

    def on_loader_start(self, state: dict):
        state = super().on_loader_start(state)
        if state["num_epochs"] - state["epoch"] <= self.no_aug_epochs:
            self.is_needed = False
        return state

    def on_batch_start(self, state: dict):
        batch = state["batch"]
        if not self.is_needed:
            return state

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
        state["batch"] = batch
        return state

    def _calc_loss(self, state: dict):
        if not self.is_needed:
            return super()._calc_loss(state)

        if self.dice == 0 or self.dice == 1:
            pred = state["pred"]
            y_a = state["batch"]["targets"]
            y_b = state["batch"]["targets"][self.index]
            loss = self.lam * self.criterion(pred, y_a) + \
                (1 - self.lam) * self.criterion(pred, y_b)
            return loss
        else:
            return super()._calc_loss(state)


def run_callbacks(callbacks, state: dict, phase: str):
    for callback in callbacks:
        state = callback.__getattribute__(phase)(state)
    return state


def train_one_epoch(model, criterion, optimizer, scheduler, loader,
                    current_epoch, batch_callbacks, num_epochs: int):
    model.train()
    state = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loader": loader,
        "epoch": current_epoch,
        "num_epochs": num_epochs
    }
    state = run_callbacks(batch_callbacks, state, "on_loader_start")

    for loader_output in progress_bar(loader, leave=False):
        state["batch"] = loader_output
        state = run_callbacks(batch_callbacks, state, "on_batch_start")
        state["pred"] = model(state["batch"]["images"])
        state = run_callbacks(batch_callbacks, state, "on_batch_end")
    return state


def train(model, criterion, optimizer, scheduler, loader, callbacks,
          num_epochs: int):
    for epoch in range(num_epochs):
        print(f"Epoch: [{epoch + 1}/{num_epochs}]:", end=" ")
        state = train_one_epoch(model, criterion, optimizer, scheduler, loader,
                                epoch + 1, callbacks, num_epochs)
        print(f"avg_loss: {state['avg_loss']:.5f}")
    return model
