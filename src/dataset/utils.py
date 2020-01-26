import gc

import numpy as np
import pandas as pd

from pathlib import Path

from src.constants import HEIGHT, WIDTH


def prepare_images(data_dir: Path, data_type: str = "train"):
    assert data_type in ["train", "test"]
    images_df_list = [
        pd.read_parquet(data_dir / f"{data_type}_image_data_{i}.parquet")
        for i in range(4)
    ]
    images = [
        df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
        for df in images_df_list
    ]

    del images_df_list
    gc.collect()

    return np.concatenate(images, axis=0)
