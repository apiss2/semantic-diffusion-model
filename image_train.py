import json
from pathlib import Path
from datetime import datetime
import torch
from diffusers.schedulers import DDPMScheduler

from config import Config
from guided_diffusion.image_datasets import create_dataloader
from guided_diffusion.model import UNetModel
from guided_diffusion.train_util import TrainLoop


def main():
    assert torch.cuda.is_available()
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
    data = create_dataloader(**config.dataset.model_dump())

    # 学習
    TrainLoop(
        model=model,
        scheduler=scheduler,
        data=data,
        save_dir=save_dir,
        **config.train.model_dump(),
    ).run_loop()


if __name__ == "__main__":
    main()
