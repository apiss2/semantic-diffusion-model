import json
from pathlib import Path

import torch
from diffusers.schedulers import DDPMScheduler

from config import Config
from guided_diffusion.image_datasets import create_dataloader
from guided_diffusion.model import UNetModel
from guided_diffusion.train_util import TrainLoop


def main():
    assert torch.cuda.is_available()
    config = Config()

    model = create_model(**config.model.model_dump())
    scheduler = DDPMScheduler(**config.scheduler.model_dump())
    model.to("cuda")

    Path("./results/config.json").write_text(json.dumps(config.model_dump(), indent=2))
    data = create_dataloader(**config.dataset.model_dump())

    TrainLoop(
        model=model,
        scheduler=scheduler,
        data=data,
        **config.train.model_dump(),
    ).run_loop()


def create_model(
    image_size,
    num_classes,
    num_channels,
    num_res_blocks,
    channel_mult="",
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    no_instance=False,
) -> UNetModel:
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    num_classes = num_classes if no_instance else num_classes + 1

    return UNetModel(
        image_size=image_size,
        in_channels=3,  # TODO: grayscale対応
        model_channels=num_channels,
        out_channels=6,  # learn sigmaの場合には2倍  # TODO: grayscale対応
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
    )


if __name__ == "__main__":
    main()
