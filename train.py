import argparse

import catalyst as ct
import pandas as pd
import torch

from pathlib import Path

from catalyst.dl import SupervisedRunner
from catalyst.utils import get_device

from src.callbacks import get_callbacks
from src.dataset import get_base_loader
from src.losses import get_loss
from src.models import get_model
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.transforms import get_transforms
from src.utils import load_config
from src.validation import get_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", nargs="*", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--debug", action="store_true", help="Whether to use debug mode")
    args = parser.parse_args()

    config = load_config(args.config)

    ct.utils.set_global_seed(config.seed)
    ct.utils.prepare_cudnn(deterministic=True)

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    output_root_dir = Path("output")
    output_base_dir = output_root_dir / args.config.replace(".yml",
                                                            "").split("/")[-1]
    output_base_dir.mkdir(exist_ok=True, parents=True)

    train_images_path = Path(config.data.train_images_path)

    df = pd.read_csv(config.data.train_df_path)
    splits = get_validation(df, config)

    transforms_dict = {
        phase: get_transforms(config, phase)
        for phase in ["train", "valid"]
    }

    cls_levels = {
        "grapheme": df.grapheme_root.nunique(),
        "vowel": df.vowel_diacritic.nunique(),
        "consonant": df.consonant_diacritic.nunique()
    }

    for i, (trn_idx, val_idx) in enumerate(splits):
        if i not in args.folds:
            continue

        print(f"Fold: {i}")

        output_dir = output_base_dir / f"fold{i}"
        output_dir.mkdir(exist_ok=True, parents=True)

        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)
        if args.debug:
            trn_df = trn_df.loc[:1000, :].reset_index(drop=True)
            val_df = val_df.loc[:1000, :].reset_index(drop=True)

        data_loaders = {
            phase: get_base_loader(
                df,
                train_images_path,
                phase=phase,
                size=(config.img_size, config.img_size),
                batch_size=config.train.batch_size,
                num_workers=config.num_workers,
                transforms=transforms_dict[phase])
            for phase, df in zip(["train", "valid"], [trn_df, val_df])
        }
        model = get_model(config).to(device)
        criterion = get_loss(config).to(device)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        callbacks = get_callbacks(config)

        runner = SupervisedRunner(
            device=device,
            input_key="images",
            input_target_key="targets",
            output_key="logits")
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=data_loaders,
            logdir=output_dir,
            scheduler=scheduler,
            num_epochs=config.train.num_epochs,
            callbacks=callbacks,
            main_metric=config.main_metric,
            minimize_metric=False,
            monitoring_params=None,
            verbose=False)
