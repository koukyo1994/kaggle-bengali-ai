import numpy as np
import torch

from pathlib import Path
from typing import Optional, Union

from catalyst.dl.core import Callback, CallbackOrder, RunnerState


class SaveWeightsCallback(Callback):
    def __init__(self,
                 to: Optional[Union[Path, str]] = None,
                 name: str = "",
                 is_larger_better=True,
                 main_metric="tar"):
        self.to = to
        if isinstance(self.to, str):
            self.to = Path(self.to)
        self.name = name
        self.best = -np.inf if is_larger_better else np.inf
        self.is_larger_better = is_larger_better
        self.main_metric = main_metric
        super().__init__(CallbackOrder.External)

    def on_epoch_end(self, state: RunnerState):
        val_metric = state.metrics.epoch_values["valid"][self.main_metric]
        to_save = False
        if self.is_larger_better and self.best < val_metric:
            to_save = True
            self.best = val_metric
        elif not self.is_larger_better and self.best > val_metric:
            to_save = True
            self.best = val_metric
        if to_save:
            weights = state.model.state_dict()
            epoch = state.epoch
            optimizer_state = state.optimizer.state_dict()
            state_dict = {
                "model_state_dict": weights,
                "epoch": epoch,
                "optimizer_state_dict": optimizer_state
            }

            logdir = state.logdir / "checkpoints"
            logdir.mkdir(exist_ok=True, parents=True)

            if self.name == "":
                torch.save(state_dict, logdir / "temp.pth")
            else:
                torch.save(state_dict, logdir / f"{self.name}.pth")

            if self.to is not None:
                if self.name == "":
                    torch.save(state_dict, self.to / "temp.pth")
                else:
                    torch.save(state_dict, self.to / f"{self.name}.pth")
