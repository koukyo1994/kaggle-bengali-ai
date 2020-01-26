import yaml
from easydict import EasyDict as edict


def _get_default():
    cfg = edict()

    # dataset
    cfg.dataset = edict()
    cfg.dataset.affine = False
    cfg.dataset.morphology = False

    # dataset
    cfg.data = edict()

    # model
    cfg.model = edict()

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
    cfg.transforms.HorizontalFlip = False
    cfg.transforms.VerticalFlip = False
    cfg.transforms.Noise = False
    cfg.transforms.Contrast = False
    cfg.transforms.Cutout = edict()
    cfg.transforms.Cutout.num_holes = 0

    # val
    cfg.val = edict()
    cfg.val.params = edict()

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
