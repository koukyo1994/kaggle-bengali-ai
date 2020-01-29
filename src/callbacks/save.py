import torch

from pathlib import Path
from typing import Optional

from catalyst.dl.core import Callback, CallbackOrder, RunnerState


class SaveWeightsCallback(Callback):
    def __init__(self, to: Optional[Path] = None, name: str = ""):
        self.to = to
        self.name = name
        super().__init__(CallbackOrder.External)

    def on_epoch_end(self, state: RunnerState):
        weights = state.model.state_dict()
        logdir = state.logdir / "checkpoints"
        logdir.mkdir(exist_ok=True, parents=True)
        if self.name == "":
            torch.save(weights, logdir / "temp.pth")
        else:
            torch.save(weights, logdir / f"{self.name}.pth")

        if self.to is not None:
            if self.name == "":
                torch.save(weights, self.to / "temp.pth")
            else:
                torch.save(weights, self.to / f"{self.name}.pth")
