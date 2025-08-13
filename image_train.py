import json
from pathlib import Path

import torch

from config import Config
from guided_diffusion.image_datasets import create_dataloader
from guided_diffusion.script_util import (
    create_gaussian_diffusion,
    create_model,
)
from guided_diffusion.train_util import TrainLoop


def main():
    assert torch.cuda.is_available()
    config = Config()
    config.diffusion.model_dump()

    model = create_model(**config.model.model_dump())
    diffusion = create_gaussian_diffusion(**config.diffusion.model_dump())
    model.to("cuda")

    Path("./results/config.json").write_text(json.dumps(config.model_dump(), indent=2))
    data = create_dataloader(**config.dataset.model_dump())

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        **config.train.model_dump(),
    ).run_loop()


if __name__ == "__main__":
    main()
