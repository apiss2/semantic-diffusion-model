from .gaussian_diffusion import get_named_beta_schedule
from .respace import SpacedDiffusion, space_timesteps
from .model import UNetModel


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
    use_fp16=False,
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
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
    )


def create_gaussian_diffusion(
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
) -> SpacedDiffusion:
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
    )
