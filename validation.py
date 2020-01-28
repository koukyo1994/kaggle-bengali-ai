import argparse

from src.dataset import get_loader
from src.inference import load_model, inference_loop
from src.losses import get_loss
from src.models import BengaliClassifier
from src.transforms import get_transforms
from src.utils import load_config
from src.validation import get_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    