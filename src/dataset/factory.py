import cv2
import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from pathlib import Path
from typing import Tuple, Dict

from .utils import crop_and_embed


class TrainDataset(torchdata.Dataset):
    def __init__(self,
                 image_dir: Path,
                 df: pd.DataFrame,
                 transforms,
                 size: Tuple[int, int],
                 cls_levels: Dict[str, int] = None,
                 onehot=True):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.size = size
        self.onehot = onehot
        self.cls_levels = cls_levels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_path = self.image_dir / f"{image_id}.png"

        image = cv2.imread(image_path)
        image = crop_and_embed(image, size=self.size)

        if self.transforms is not None:
            image = self.transforms(image)

        grapheme = self.df.loc[idx, "grapheme_root"]
        vowel = self.df.loc[idx, "vowel_diacritic"]
        consonant = self.df.loc[idx, "consonant_diacritic"]

        if self.onehot:
            grapheme_levels = self.cls_levels["grapheme"]
            vowel_levels = self.cls_levels["vowel"]
            consonant_levels = self.cls_levels["consonant"]
            total_n_levels = grapheme_levels + vowel_levels + consonant_levels
            label = np.zeros(total_n_levels, dtype=np.float32)
            label[grapheme] = 1.0
            label[grapheme_levels + vowel] = 1.0
            label[grapheme_levels + vowel_levels + consonant] = 1.0

        else:
            label = np.zeros(3, dtype=int)
            label[0] = grapheme
            label[1] = vowel
            label[2] = consonant
        return image, label


class TestDataset(torchdata.Dataset):
    def __init__(self, image_dir: Path, df: pd.DataFrame, transforms,
                 size: Tuple[int, int]):
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_path = self.image_dir / f"{image_id}.png"

        image = cv2.imread(image_path)
        image = crop_and_embed(image, size=self.size)

        if self.transforms is not None:
            image = self.transforms(image)
        return image


def get_loader(df_path: Path,
               image_dir: Path,
               phase: str = "train",
               size: Tuple[int, int] = (128, 128),
               batch_size=256,
               num_workers=2,
               transforms=None,
               cls_levels=None,
               onehot=None):
    assert phase in ["train", "val", "test"]
    df = pd.read_csv(df_path)
    if phase == "test":
        dataset = TestDataset(image_dir, df, transforms, size)
        is_shuffle = False
        drop_last = False
    else:
        if phase == "train":
            is_shuffle = True
            drop_last = True
        else:
            is_shuffle = False
            drop_last = False
        if onehot is not None:
            if cls_levels is None:
                raise ValueError(
                    "if 'onehot' is set to None, cls_levels must be set")
            else:
                dataset = TrainDataset(  # type: ignore
                    image_dir, df, transforms, size, cls_levels, onehot)
        else:
            dataset = TrainDataset(  # type: ignore
                image_dir, df, transforms, size)
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last)
