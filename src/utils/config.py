import yaml
from easydict import EasyDict as edict


def _get_default():
    cfg = edict()

    # dataset
    cfg.dataset = edict()
    cfg.dataset.train = edict()
    cfg.dataset.val = edict()
    cfg.dataset.test = edict()
    cfg.dataset.train.affine = False
    cfg.dataset.train.morphology = False
    cfg.dataset.val.affine = False
    cfg.dataset.val.morphology = False
    cfg.dataset.test.affine = False
    cfg.dataset.test.morphology = False

    # dataset
    cfg.data = edict()

    # model
    cfg.model = edict()
    cfg.model.model_name = "resnet18"
    cfg.model.num_classes = 186
    cfg.model.pretrained = True
    cfg.model.head = "linear"
    cfg.model.in_channels = 3
    cfg.model.outputs = ["grapheme", "vowel", "consonant"]

    # train
    cfg.train = edict()

    # test
    cfg.test = edict()

    # loss
    cfg.loss = edict()
    cfg.loss.params = edict()

    # optimizer
    cfg.optimizer = edict()
    cfg.optimizer.params = edict()

    # scheduler
    cfg.scheduler = edict()
    cfg.scheduler.params = edict()

    # transforms:
    cfg.transforms = edict()
    cfg.transforms.train = edict()
    cfg.transforms.train.HorizontalFlip = False
    cfg.transforms.train.VerticalFlip = False
    cfg.transforms.train.Noise = False
    cfg.transforms.train.Contrast = False
    cfg.transforms.train.Rotate = False
    cfg.transforms.train.RandomScale = False
    cfg.transforms.train.Cutout = edict()
    cfg.transforms.train.Cutout.num_holes = 0
    cfg.transforms.val = edict()
    cfg.transforms.val.HorizontalFlip = False
    cfg.transforms.val.VerticalFlip = False
    cfg.transforms.val.Noise = False
    cfg.transforms.val.Contrast = False
    cfg.transforms.val.Rotate = False
    cfg.transforms.val.RandomScale = False
    cfg.transforms.val.Cutout = edict()
    cfg.transforms.val.Cutout.num_holes = 0
    cfg.transforms.test = edict()
    cfg.transforms.test.HorizontalFlip = False
    cfg.transforms.test.VerticalFlip = False
    cfg.transforms.test.Noise = False
    cfg.transforms.test.Contrast = False
    cfg.transforms.test.Rotate = False
    cfg.transforms.test.RandomScale = False
    cfg.transforms.test.Cutout = edict()
    cfg.transforms.test.Cutout.num_holes = 0
    cfg.transforms.mean = [0.485, 0.456, 0.406]
    cfg.transforms.std = [0.229, 0.224, 0.225]

    # val
    cfg.val = edict()
    cfg.val.params = edict()

    cfg.callbacks = []

    return cfg


def _merge_config(src: edict, dst: edict):
    if not isinstance(src, edict):
        return
    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, "r") as f:
        config = edict(yaml.load(f, Loader=yaml.SafeLoader))

    cfg = _get_default()
    _merge_config(config, cfg)
    return cfg


def save_config(config: edict, fname):
    with open(fname, "w") as f:
        yaml.dump(config, f)
