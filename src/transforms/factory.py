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
    if cfg.ShiftScaleRotate:
        list_transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0, rotate_limit=15, p=0.5))
    if cfg.RandomResizedCrop:
        list_transforms.append(
            A.RandomResizedCrop(128, 128, scale=(0.8, 1), p=0.5))
    if cfg.CoarseDropout:
        list_transforms.append(
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2))
    if cfg.GridDistortion:
        list_transforms.append(A.GridDistortion(p=0.2))

    list_transforms.append(
        A.Normalize(
            mean=config.transforms.mean,
            std=config.transforms.std,
            p=1,
            always_apply=True))

    return A.Compose(list_transforms, p=1.0)
