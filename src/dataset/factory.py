import cv2
import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from pathlib import Path
from typing import Tuple, Dict

from .utils import (crop_and_embed, normalize, affine_image,
                    random_erosion_or_dilation, to_image, crop_resize,
                    binarization_and_opening, whiten_background)


class BaseDataset(torchdata.Dataset):
    def __init__(self,
                 image_dir: Path,
                 df: pd.DataFrame,
                 transforms,
                 size: Tuple[int, int],
                 binarization=False,
                 whitening=False):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.size = size
        self.binarization = binarization
        self.whitening = whitening

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_path = self.image_dir / f"{image_id}.png"

        image = cv2.imread(str(image_path))
        if self.binarization:
            image = binarization_and_opening(image)

        if self.whitening:
            image = whiten_background(image)
        longer_side = image.shape[1]
        if image.ndim == 2:
            new_image = np.ones(
                (longer_side, longer_side), dtype=np.uint8) * 255
        else:
            new_image = np.ones(
                (longer_side, longer_side, 3), dtype=np.uint8) * 255
        offset = np.random.randint(0, longer_side - image.shape[0])
        new_image[offset:offset + image.shape[0], :] = image

        if self.transforms is not None:
            image = self.transforms(image=new_image)["image"]
        image = cv2.resize(image, self.size)
        if image.shape[2] == 3:
            image = np.moveaxis(image, -1, 0)
        grapheme = self.df.loc[idx, "grapheme_root"]
        vowel = self.df.loc[idx, "vowel_diacritic"]
        consonant = self.df.loc[idx, "consonant_diacritic"]
        label = np.zeros(3, dtype=int)
        label[0] = grapheme
        label[1] = vowel
        label[2] = consonant
        return {"images": image, "targets": label}


class BaseTestDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 transforms,
                 size: Tuple[int, int],
                 binarization=False,
                 whitening=False):
        self.images = df.iloc[:, 1:].values.reshape(-1, 137, 236)
        self.size = size
        self.transforms = transforms
        self.binarization = binarization
        self.whitening = whitening

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if image.ndim == 2:
            image = np.moveaxis(np.stack([image, image, image]), 0, -1)
        if self.binarization:
            image = binarization_and_opening(image)
        if self.whitening:
            image = whiten_background(image)
        longer_side = image.shape[1]
        new_image = np.ones((longer_side, longer_side), dtype=np.uint8) * 255
        offset = np.random.randint(0, longer_side - image.shape[0])
        new_image[offset:offset + image.shape[0], :] = image
        if self.transforms is not None:
            image = self.transforms(image=new_image)["image"]
        image = cv2.resize(image, self.size)
        if image.shape[2] == 3:
            image = np.moveaxis(image, -1, 0)
        return image


class TrainDataset(torchdata.Dataset):
    def __init__(self,
                 image_dir: Path,
                 df: pd.DataFrame,
                 transforms,
                 size: Tuple[int, int],
                 cls_levels: Dict[str, int] = None,
                 affine=True,
                 morphology=True,
                 onehot=True):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.size = size
        self.onehot = onehot
        self.cls_levels = cls_levels
        self.affine = affine
        self.morphology = morphology

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_path = self.image_dir / f"{image_id}.png"

        image = cv2.imread(str(image_path))
        image = normalize(image)
        image = crop_and_embed(image, size=self.size, threshold=5. / 255.)
        if self.affine:
            image = affine_image(image)
        if self.morphology:
            image = random_erosion_or_dilation(image)
        image = to_image(image)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        if image.shape[2] == 3:
            image = np.moveaxis(image, -1, 0)
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
        return {"images": image, "targets": label}


class TestDataset(torchdata.Dataset):
    def __init__(self,
                 image_dir: Path,
                 df: pd.DataFrame,
                 transforms,
                 size: Tuple[int, int],
                 affine=True,
                 morphology=True):
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms
        self.size = size
        self.affine = affine
        self.morphology = morphology

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_path = self.image_dir / f"{image_id}.png"

        image = cv2.imread(image_path)
        image = normalize(image)
        image = crop_and_embed(image, size=self.size, threshold=5. / 255.)
        if self.affine:
            image = affine_image(image)
        if self.morphology:
            image = random_erosion_or_dilation(image)
        image = to_image(image)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        if image.shape[2] == 3:
            image = np.moveaxis(image, -1, 0)
        return image


def get_base_test_loader(df: pd.DataFrame,
                         size: Tuple[int, int] = (128, 128),
                         batch_size=256,
                         num_workers=2,
                         transforms=None,
                         binarization=False,
                         whitening=False):
    dataset = BaseTestDataset(df, transforms, size, binarization, whitening)
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)


def get_base_loader(df: pd.DataFrame,
                    image_dir: Path,
                    phase: str = "train",
                    size: Tuple[int, int] = (128, 128),
                    batch_size=256,
                    num_workers=2,
                    transforms=None,
                    binarization=False,
                    whitening=False):
    assert phase in ["train", "valid"]
    if phase == "train":
        is_shuffle = True
        drop_last = True
    else:
        is_shuffle = False
        drop_last = False

    dataset = BaseDataset(  # type: ignore
        image_dir, df, transforms, size, binarization, whitening)
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last)


def get_loader(df: pd.DataFrame,
               image_dir: Path,
               phase: str = "train",
               size: Tuple[int, int] = (128, 128),
               batch_size=256,
               num_workers=2,
               transforms=None,
               cls_levels=None,
               affine=True,
               morphology=True,
               onehot=None):
    assert phase in ["train", "valid", "test"]
    if phase == "test":
        dataset = TestDataset(image_dir, df, transforms, size, affine,
                              morphology)
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
                    image_dir,
                    df,
                    transforms,
                    size,
                    cls_levels,
                    affine=affine,
                    morphology=morphology,
                    onehot=onehot)
        else:
            dataset = TrainDataset(  # type: ignore
                image_dir,
                df,
                transforms,
                size,
                affine=affine,
                morphology=morphology)
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last)
