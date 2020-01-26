from pathlib import Path

from src.dataset.utils import prepare_images
from src.utils import timer

if __name__ == "__main__":
    with timer(name="load data"):
        images = prepare_images(
            Path("input/bengaliai-cv19"), data_type="train")

    import pdb
    pdb.set_trace()
