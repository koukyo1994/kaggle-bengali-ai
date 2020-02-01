import albumentations as A

from easydict import EasyDict as edict


def get_transforms(config: edict, phase: str = "train"):
    assert phase in ["train", "valid", "test"]
    if phase == "train":
        cfg = config.transforms.train
    elif phase == "valid":
        cfg = config.transforms.val
    elif phase == "test":
        cfg = config.transforms.test
    list_transforms = []
    if cfg.HorizontalFlip:
        list_transforms.append(A.HorizontalFrip())
    if cfg.VerticalFlip:
        list_transforms.append(A.VerticalFlip())
    if cfg.Rotate:
        list_transforms.append(A.Rotate(limit=15))
    if cfg.RandomScale:
        list_transforms.append(A.RandomScale())
    if cfg.Noise:
        list_transforms.append(
            A.OneOf(
                [A.GaussNoise(), A.IAAAdditiveGaussianNoise()], p=0.5))
    if cfg.Contrast:
        list_transforms.append(
            A.OneOf(
                [A.RandomContrast(0.5),
                 A.RandomGamma(),
                 A.RandomBrightness()],
                p=0.5))
    if cfg.Cutout.num_holes > 0:
        list_transforms.append(A.Cutout(**cfg.Cutout))

    list_transforms.append(
        A.Normalize(
            mean=config.transforms.mean,
            std=config.transforms.std,
            p=1,
            always_apply=True))

    return A.Compose(list_transforms, p=1.0)
