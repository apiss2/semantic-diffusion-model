from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


class DatasetConfig(BaseModel):
    # 複数の設定に跨る値を定義
    image_size: int | None = None
    batch_size: int | None = None
    num_classes: int | None = None
    grayscale: bool | None = None
    # 他設定の定義
    image_dir: str = "D:/Datasets/CelebAMask-HQ/CelebA-HQ-img-512/"
    label_dir: str = "D:/Datasets/CelebAMask-HQ/CelebAMask-HQ-mask-img/"
    num_workers: int = 8


class TrainConfig(BaseModel):
    # 複数の設定に跨る値を定義
    batch_size: int | None = None
    num_classes: int | None = None
    use_fp16: bool | None = None
    grayscale: bool | None = None
    # 他設定の定義
    lr: float = 1e-4
    ema_rate: str = "0.9,0.99"  # comma-separated list of EMA values
    weight_decay: float = 1e-3
    lr_anneal_steps: int = 50000
    save_interval: int = 1000
    resume_checkpoint: str = ""
    fp16_scale_growth: float = 1e-2
    drop_rate: float = 0.0


class ModelConfig(BaseModel):
    # 複数の設定に跨る値を定義
    image_size: int | None = None
    num_classes: int | None = None
    use_fp16: bool | None = None
    # 他設定の定義
    num_channels: int = 128
    num_res_blocks: int = 2
    num_heads: int = 1
    num_head_channels: int = 64
    num_heads_upsample: int = -1
    attention_resolutions: str = "32,16,8"
    channel_mult: str = ""
    dropout: float = 0.0
    class_cond: bool = True
    use_checkpoint: bool = True
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    no_instance: bool = True


class DiffusionConfig(BaseModel):
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = ""


class Config(BaseSettings):
    # 複数の設定に跨る値を定義
    image_size: int = Field(128)
    batch_size: int = Field(4)
    num_classes: int = Field(19)
    use_fp16: bool = Field(True)
    grayscale: bool = Field(False)

    # その他定義
    schedule_sampler: str = "uniform"

    train: TrainConfig = TrainConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()

    class Config:
        env_nested_delimiter = "__"

    @model_validator(mode="after")
    def _sync(self):
        """
        共有フィールドの同期ポリシー
        - 親の値(self.<key>)を唯一の真実とする
        - サブ設定の値が None なら親を注入
        - サブ設定の値が親と異なれば明示エラー
        """
        shared_map = {
            "image_size": ("dataset", "model"),
            "batch_size": ("dataset", "train"),
            "num_classes": ("dataset", "train", "model"),
            "use_fp16": ("train", "model"),
            "grayscale": ("dataset", "train"),
        }

        for key, sections in shared_map.items():
            top_val = getattr(self, key)
            for sec in sections:
                sub_cfg = getattr(self, sec)
                sub_val = getattr(sub_cfg, key, None)

                if sub_val is None:
                    setattr(sub_cfg, key, top_val)
                elif sub_val != top_val:
                    raise ValueError(f"{sec}.{key}({sub_val}) != {key}({top_val})")
        return self
