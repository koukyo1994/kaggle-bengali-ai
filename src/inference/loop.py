import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from typing import Optional

from catalyst.utils import get_device
from fastprogress import progress_bar


def inference_loop(model: nn.Module,
                   loader: torchdata.DataLoader,
                   cls_levels: dict,
                   loss_fn: Optional[nn.Module] = None,
                   requires_soft=False):
    n_grapheme = cls_levels["grapheme"]
    n_vowel = cls_levels["vowel"]
    n_consonant = cls_levels["consonant"]

    dataset_length = len(loader.dataset)
    prediction = np.zeros((dataset_length, 3), dtype=np.uint8)
    if requires_soft:
        soft_prediction = np.zeros(
            (dataset_length, n_grapheme + n_vowel + n_consonant),
            dtype=np.float32)

    batch_size = loader.batch_size
    device = get_device()

    avg_loss = 0.
    model.eval()

    targets: Optional[torch.Tensor] = None

    for i, batch in enumerate(progress_bar(loader, leave=False)):
        with torch.no_grad():
            if isinstance(batch, dict):
                images = batch["images"].to(device)
                targets = batch["targets"].to(device)
            else:
                images = batch.to(device)
                targets = None
            pred = model(images).detach()
            if loss_fn is not None and targets is not None:
                avg_loss += loss_fn(
                    pred, batch["targets"].to(device)).item() / len(loader)
            head = 0
            tail = n_grapheme
            pred_grapheme = torch.argmax(
                pred[:, head:tail], dim=1).cpu().numpy()

            head = tail
            tail = head + n_vowel
            pred_vowel = torch.argmax(pred[:, head:tail], dim=1).cpu().numpy()

            head = tail
            tail = head + n_consonant
            pred_consonant = torch.argmax(
                pred[:, head:tail], dim=1).cpu().numpy()

            prediction[i * batch_size:(i + 1) * batch_size, 0] = pred_grapheme
            prediction[i * batch_size:(i + 1) * batch_size, 1] = pred_vowel
            prediction[i * batch_size:(i + 1) * batch_size, 2] = pred_consonant

            if requires_soft:
                head = 0
                tail = n_grapheme
                soft_prediction[i * batch_size:(i + 1) *
                                batch_size, head:tail] = F.softmax(
                                    pred[:, head:tail], dim=1).cpu().numpy()

                head = tail
                tail = head + n_vowel
                soft_prediction[i * batch_size:(i + 1) *
                                batch_size, head:tail] = F.softmax(
                                    pred[:, head:tail], dim=1).cpu().numpy()

                head = tail
                tail = head + n_consonant
                soft_prediction[i * batch_size:(i + 1) *
                                batch_size, head:tail] = F.softmax(
                                    pred[:, head:tail], dim=1).cpu().numpy()

    return_dict = {"prediction": prediction, "loss": avg_loss}
    if requires_soft:
        return_dict["soft_prediction"] = soft_prediction

    return return_dict
