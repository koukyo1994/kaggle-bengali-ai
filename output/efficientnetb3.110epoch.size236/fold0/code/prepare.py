import pandas as pd

from PIL import Image
from pathlib import Path

from fastprogress import master_bar, progress_bar

from src.constants import HEIGHT, WIDTH

if __name__ == "__main__":
    data_dir = Path("input/bengaliai-cv19")

    mb = master_bar(["train", "test"])
    for data_type in mb:
        output_dir = data_dir / f"{data_type}_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in progress_bar(range(4), parent=mb, leave=False):
            df = pd.read_parquet(
                data_dir / f"{data_type}_image_data_{i}.parquet")
            images = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

            for im in range(len(images)):
                image = Image.fromarray(images[im])
                image_id = df.loc[im, "image_id"]
                image.save(output_dir / f"{image_id}.png")
