import albumentations as A

from easydict import EasyDict as edict


def get_transforms(config: edict):
    list_transforms = []
    if config.transforms.HorizontalFlip:
        list_transforms.append(A.HorizontalFrip())
    if config.transforms.VerticalFlip:
        list_transforms.append(A.VerticalFlip())
    if config.transforms.Rotate:
        list_transforms.append(A.Rotate(limit=15))
    if config.transforms.RandomScale:
        list_transforms.append(A.RandomScale())
    if config.transforms.Noise:
        list_transforms.append(
            A.OneOf(
                [A.GaussNoise(), A.IAAAdditiveGaussianNoise()], p=0.5))
    if config.transforms.Contrast:
        list_transforms.append(
            A.OneOf(
                [A.RandomContrast(0.5),
                 A.RandomGamma(),
                 A.RandomBrightness()],
                p=0.5))
    if config.transforms.Cutout.num_holes > 0:
        list_transforms.append(A.Cutout(**config.Cutout))

    list_transforms.append(
        A.Normalize(
            mean=config.transforms.mean, std=config.transforms.std, p=1))

    return A.Compose(list_transforms, p=1.0)
