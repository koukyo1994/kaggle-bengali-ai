import albumentations as A

from easydict import EasyDict as edict


def get_transforms(config: edict):
    list_transforms = []
    if config.transforms.HorizontalFlip:
        list_transforms.append(A.HorizontalFrip())
    if config.transforms.VerticalFlip:
        list_transforms.append(A.VerticalFlip())
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

    return A.Compose(list_transforms, p=1.0)
