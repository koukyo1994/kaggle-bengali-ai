import argparse

import catalyst as ct
import pandas as pd

from pathlib import Path

from catalyst.dl import SupervisedRunner

from src.callbacks import MacroAverageRecall, SaveWeightsCallback
from src.dataset import get_loader
from src.losses import get_loss
from src.models import BengaliClassifier
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.transforms import get_transforms
from src.utils import load_config
from src.validation import get_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    ct.utils.set_global_seed(config.seed)
    ct.utils.prepare_cudnn(deterministic=True)

    output_root_dir = Path("output")
    output_base_dir = output_root_dir / args.config.replace(".yml",
                                                            "").split("/")[-1]
    output_base_dir.mkdir(exist_ok=True, parents=True)

    train_images_path = Path(config.data.train_images_path)

    df = pd.read_csv(config.data.train_df_path)
    splits = get_validation(df, config)

    transforms = get_transforms(config)

    cls_levels = {
        "grapheme": df.grapheme_root.nunique(),
        "vowel": df.vowel_diacritic.nunique(),
        "consonant": df.consonant_diacritic.nunique()
    }

    for i, (trn_idx, val_idx) in enumerate(splits):
        print(f"Fold: {i}")

        output_dir = output_base_dir / f"fold{i}"
        output_dir.mkdir(exist_ok=True, parents=True)

        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)
        data_loaders = {
            phase: get_loader(
                df,
                train_images_path,
                phase=phase,
                size=(config.img_size, config.img_size),
                batch_size=config.train.batch_size,
                num_workers=config.num_workers,
                transforms=transforms,
                cls_levels=cls_levels,
                affine=config.dataset.train.affine
                if phase == "train" else config.dataset.val.affine,
                morphology=config.dataset.train.morphology
                if phase == "train" else config.dataset.val.morphology,
                onehot=True if config.loss.name == "bce" else False)
            for phase, df in zip(["train", "valid"], [trn_df, val_df])
        }
        model = BengaliClassifier(**config.model)
        criterion = get_loss(config)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        callbacks = [
            MacroAverageRecall(
                n_grapheme=cls_levels["grapheme"],
                n_vowel=cls_levels["vowel"],
                n_consonant=cls_levels["consonant"],
                loss_type=config.loss.name),
            SaveWeightsCallback(
                to=Path(config.checkpoints
                        ) if config.checkpoints is not None else None)
        ]

        runner = SupervisedRunner(
            device=ct.utils.get_device(),
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
            main_metric="mar",
            minimize_metric=False,
            monitoring_params=None,
            verbose=True)
