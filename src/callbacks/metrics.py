import numpy as np
import torch

from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from sklearn.metrics import recall_score


class MacroAverageRecall(Callback):
    def __init__(self,
                 n_grapheme=168,
                 n_vowel=11,
                 n_consonant=7,
                 loss_type: str = "bce",
                 prefix: str = "mar",
                 output_key: str = "logits",
                 target_key: str = "targets"):
        self.prefix = prefix
        self.output_key = output_key
        self.target_key = target_key
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.loss_type = loss_type
        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: RunnerState):
        targ = state.input[self.target_key].detach()
        out = state.output[self.output_key]
        head = 0
        tail = self.n_grapheme
        grapheme = torch.sigmoid(out[:, head:tail])
        grapheme_np = torch.argmax(grapheme, dim=1).detach().cpu().numpy()
        if self.loss_type == "bce":
            grapheme_target = torch.argmax(
                targ[:, head:tail], dim=1).cpu().numpy()
        else:
            grapheme_target = targ[:, 0].cpu().numpy()

        head = tail
        tail = head + self.n_vowel
        vowel = torch.sigmoid(out[:, head:tail])
        vowel_np = torch.argmax(vowel, dim=1).detach().cpu().numpy()
        if self.loss_type == "bce":
            vowel_target = torch.argmax(
                targ[:, head:tail], dim=1).cpu().numpy()
        else:
            vowel_target = targ[:, 1].cpu().numpy()

        head = tail
        tail = head + self.n_consonant
        consonant = torch.sigmoid(out[:, head:tail])
        consonant_np = torch.argmax(consonant, dim=1).detach().cpu().numpy()
        if self.loss_type == "bce":
            consonant_target = torch.argmax(
                targ[:, head:tail], dim=1).cpu().numpy()
        else:
            consonant_target = targ[:, 2].cpu().numpy()

        scores = []
        scores.append(
            recall_score(
                grapheme_target, grapheme_np, average="macro",
                zero_division=0))
        scores.append(
            recall_score(
                vowel_target, vowel_np, average="macro", zero_division=0))
        scores.append(
            recall_score(
                consonant_target,
                consonant_np,
                average="macro",
                zero_division=0))
        final_score = np.average(scores, weights=[2, 1, 1])
        state.metrics.add_batch_value(name=self.prefix, value=final_score)


class AverageRecall(Callback):
    def __init__(self,
                 index: int,
                 offset: int,
                 n_classes: int,
                 prefix: str,
                 loss_type: str = "bce",
                 output_key: str = "logits",
                 target_key: str = "targets"):
        self.index = index
        self.offset = offset
        self.n_classes = n_classes
        self.prefix = prefix
        self.loss_type = loss_type
        self.output_key = output_key
        self.target_key = target_key
        self.recall = 0.0
        super().__init__(CallbackOrder.Metric)

    def on_epoch_start(self, state: RunnerState):
        self.x = torch.arange(0, self.n_classes)
        self.cm = torch.zeros((self.n_classes, self.n_classes))

    def on_batch_end(self, state: RunnerState):
        targ = state.input[self.target_key].detach()
        out = state.output[self.output_key].detach()
        head = self.offset
        tail = self.offset + self.n_classes
        if self.loss_type == "bce":
            pred = torch.argmax(torch.sigmoid(out[:, head:tail]), dim=1).cpu()
            pred_np = pred.numpy()
            target = torch.argmax(targ[:, head:tail], dim=1).cpu()
            target_np = target.numpy()
        else:
            pred = torch.argmax(out[:, head:tail], dim=1).cpu()
            pred_np = pred.numpy()
            target = targ[:, self.index].cpu()
            target_np = target.numpy()
        cm = ((pred == self.x[:, None]) & (target == self.x[:, None, None])) \
            .sum(dim=2, dtype=torch.float32)
        self.cm += cm
        score = recall_score(
            target_np, pred_np, average="macro", zero_devision=0)
        state.metrics.add_batch_value(name="batch_" + self.prefix, value=score)

    def _recall(self):
        rec = torch.diag(self.cm) / (self.cm.sum(dim=1) + 1e-9)
        return rec

    def on_loader_end(self, state: RunnerState):
        self.recall = self._recall()


class TotalAverageRecall(Callback):
    def __init__(self,
                 n_grapheme=168,
                 n_vowel=11,
                 n_consonant=7,
                 loss_type: str = "bce",
                 prefix: str = "tar",
                 output_key: str = "logits",
                 target_key: str = "targets"):
        self.prefix = prefix
        self.grapheme_callback = AverageRecall(
            index=0,
            offset=0,
            n_classes=n_grapheme,
            prefix="grapheme_recall",
            loss_type=loss_type,
            output_key=output_key,
            target_key=target_key)
        self.vowel_callback = AverageRecall(
            index=1,
            offset=n_grapheme,
            n_classes=n_vowel,
            prefix="vowel_recall",
            loss_type=loss_type,
            output_key=output_key,
            target_key=target_key)
        self.consonant_callback = AverageRecall(
            index=2,
            offset=n_grapheme + n_vowel,
            n_classes=n_consonant,
            prefix="consonant_recall",
            loss_type=loss_type,
            output_key=output_key,
            target_key=target_key)
        super().__init__(CallbackOrder.Metric)

    def on_epoch_start(self, state):
        self.grapheme_callback.on_epoch_start(state)
        self.vowel_callback.on_epoch_start(state)
        self.consonant_callback.on_epoch_start(state)

    def on_batch_end(self, state):
        self.grapheme_callback.on_batch_end(state)
        self.vowel_callback.on_batch_end(state)
        self.consonant_callback.on_batch_end(state)

    def on_loader_end(self, state: RunnerState):
        self.grapheme_callback.on_loader_end(state)
        self.vowel_callback.on_loader_end(state)
        self.consonant_callback.on_loader_end(state)
        final_score = np.average([
            self.grapheme_callback.recall, self.vowel_callback.recall,
            self.consonant_callback.recall
        ],
                                 weights=[2, 1, 1])
        state.metrics.add_batch_value(name=self.prefix, value=final_score)
