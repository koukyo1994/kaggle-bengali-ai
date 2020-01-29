import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from easydict import EasyDict as edict

from src.dataset import get_base_loader
from src.inference import load_model, inference_loop, macro_average_recall
from src.losses import get_loss
from src.transforms import get_transforms
from src.utils import load_config, save_json
from src.validation import get_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--bin_name", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    output_root_dir = Path("output")
    output_base_dir = output_root_dir / args.config.replace(".yml",
                                                            "").split("/")[-1]

    valid_images_path = Path(config.data.train_images_path)
    df = pd.read_csv(config.data.train_df_path)
    splits = get_validation(df, config)

    transforms_dict = {"valid": get_transforms(config, "valid")}
    cls_levels = {
        "grapheme": df.grapheme_root.nunique(),
        "vowel": df.vowel_diacritic.nunique(),
        "consonant": df.consonant_diacritic.nunique()
    }

    if len(args.folds) == 0:
        folds = [0]
    else:
        folds = args.folds

    oof_preds = np.zeros((len(df), 3))

    for i in folds:
        print(f"Fold: {i}")
        _, val_idx = splits[i]

        checkpoints_path = output_base_dir / f"fold{i}/checkpoints"

        val_df = df.loc[val_idx, :].reset_index(drop=True)
        data_loader = get_base_loader(
            val_df,
            valid_images_path,
            phase="valid",
            size=(config.img_size, config.img_size),
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
            transforms=transforms_dict["valid"])
        model = load_model(config, checkpoints_path / args.bin_name)
        criterion = get_loss(config)

        prediction = inference_loop(
            model, data_loader, cls_levels, criterion, requires_soft=False)
        score = macro_average_recall(prediction["prediction"], val_df)
        oof_preds[val_idx, :] = prediction["prediction"]

        config.eval_result = edict()
        config.eval_result[f"fold{i}"] = edict()
        config.eval_result[f"fold{i}"].score = score
        config.eval_result[f"fold{i}"].loss = prediction["loss"]

    save_json(config, output_base_dir / "result.json")
    np.save(output_base_dir / "oof_preds.npy", oof_preds)
