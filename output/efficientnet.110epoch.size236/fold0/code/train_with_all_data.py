import argparse

import pandas as pd

from pathlib import Path

from src.dataset import get_base_loader
from src.losses import get_loss
from src.models import get_model
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.transforms import get_transforms
from src.utils import load_config, seed_torch
from src.trainers.trainer import MixupOrCutmixCallback, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--debug", action="store_true", help="Whether to use debug mode")
    args = parser.parse_args()

    config = load_config(args.config)

    seed_torch(config.seed)

    output_root_dir = Path("output")
    output_base_dir = output_root_dir / args.config.replace(".yml",
                                                            "").split("/")[-1]
    output_base_dir.mkdir(exist_ok=True, parents=True)

    train_images_path = Path(config.data.train_images_path)

    df = pd.read_csv(config.data.train_df_path)
    transforms = get_transforms(config, phase="train")

    cls_levels = {
        "grapheme": df.grapheme_root.nunique(),
        "vowel": df.vowel_diacritic.nunique(),
        "consonant": df.consonant_diacritic.nunique()
    }

    if args.debug:
        trn_df = df.loc[:1000, :].reset_index(drop=True)
    else:
        trn_df = df

    data_loader = get_base_loader(
        trn_df,
        train_images_path,
        phase="train",
        size=(config.img_size, config.img_size),
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        transforms=transforms)
    model = get_model(config)
    criterion = get_loss(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    callbacks = [
        MixupOrCutmixCallback(
            criterion, mixup_prob=0.5, cutmix_prob=0.0, no_aug_epochs=5)
    ]
    model = train(model, criterion, optimizer, scheduler, data_loader,
                  callbacks, config["train"]["num_epochs"])
