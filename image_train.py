import argparse

import torch
from guided_diffusion.image_datasets import create_dataloader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    assert torch.cuda.is_available()
    args = create_argparser().parse_args()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to("cuda")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    data = create_dataloader(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        size=args.image_size,
        num_classes=19,
        batch_size=args.batch_size,
        num_workers=8,
    )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        drop_rate=args.drop_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        image_dir="D:/Datasets/CelebAMask-HQ/CelebA-HQ-img-512/",
        label_dir="D:/Datasets/CelebAMask-HQ/CelebAMask-HQ-mask-img/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        is_train=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
