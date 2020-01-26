import numpy as np
import torch

from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from sklearn.metrics import recall_score


class MacroAverageRecall(Callback):
    def __init__(self,
                 n_grapheme=168,
                 n_vowel=11,
                 n_consonant=7,
                 prefix: str = "mar",
                 output_key: str = "logits",
                 target_key: str = "targets"):
        self.prefix = prefix
        self.output_key = output_key
        self.target_key = target_key
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: RunnerState):
        targ = state.input[self.target_key].detach()
        out = state.output[self.output_key]
        head = 0
        tail = self.n_grapheme
        grapheme = torch.sigmoid(out[:, head:tail])
        grapheme_np = torch.argmax(grapheme, dim=1).detach().cpu().numpy()
        grapheme_target = torch.argmax(targ[:, head:tail], dim=1).cpu().numpy()

        head = tail
        tail = head + self.n_vowel
        vowel = torch.sigmoid(out[:, head:tail])
        vowel_np = torch.argmax(vowel, dim=1).detach().cpu().numpy()
        vowel_target = torch.argmax(targ[:, head:tail], dim=1).cpu().numpy()

        head = tail
        tail = head + self.n_consonant
        consonant = torch.sigmoid(out[:, head:tail])
        consonant_np = torch.argmax(consonant, dim=1).detach().cpu().numpy()
        consonant_target = torch.argmax(
            targ[:, head:tail], dim=1).cpu().numpy()

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
