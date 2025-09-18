import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import albumentations as A
import torch
from diffusers.schedulers import DDPMScheduler

from config import Config
from guided_diffusion.image_datasets import create_dataloader
from guided_diffusion.model import UNetModel
from guided_diffusion.train_util import TrainLoop


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main():
    assert torch.cuda.is_available()
    args = get_args()
    if args.config:
        config = Config.from_json(args.config)
    else:
        config = Config()

    # 保存先の作成
    if config.train.resume_checkpoint == "":
        save_dir_root = Path(config.save_dir_root)
        save_dir_name = datetime.now().strftime("%Y-%m%d-%H%M")
        save_dir = save_dir_root / save_dir_name
        save_dir.mkdir(exist_ok=True)
        # configの保存
        save_path = save_dir / "config.json"
        save_path.write_text(json.dumps(config.model_dump(), indent=2))
    else:
        # save_dir はckptファイルの2階層上。(save_dir/ckpt/model.pt)
        save_dir = Path(config.train.resume_checkpoint).parent.parent

    # モデル等の定義
    model = UNetModel(**config.model.model_dump()).to("cuda")
    scheduler = DDPMScheduler(**config.scheduler.model_dump())
    kwargs = config.dataset.model_dump()
    kwargs.update(transforms=build_transforms(config))
    data = create_dataloader(**kwargs)

    # 学習
    TrainLoop(
        model=model,
        scheduler=scheduler,
        data=data,
        save_dir=save_dir,
        **config.train.model_dump(),
    ).run_loop()


def build_transforms(config: Config) -> A.Compose:
    def _build_transform(t) -> A.BasicTransform:
        if not hasattr(A, t.name):
            raise ValueError(f"{t.name} is not in Albumentations")
        Tfm = getattr(A, t.name)
        if t.transforms is not None:
            return Tfm([_build_transform(child) for child in t.transforms], **t.kwargs)
        else:
            return Tfm(**t.kwargs)

    return A.Compose([_build_transform(t) for t in config.dataset.transforms])


if __name__ == "__main__":
    main()
