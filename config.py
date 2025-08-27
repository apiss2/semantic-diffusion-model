from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    # 複数の設定に跨る値を定義
    image_size: int | None = None
    batch_size: int | None = None
    num_classes: int | None = None
    grayscale: bool | None = None
    # 他設定の定義
    image_dir: str | None = None
    label_dir: str | None = None
    conds_json: str | None = None
    num_workers: int = 8
    image_suffix: str = ".jpg"
    label_suffix: str = ".png"


class TrainConfig(BaseModel):
    # 複数の設定に跨る値を定義
    batch_size: int | None = None
    num_classes: int | None = None
    grayscale: bool | None = None
    # 他設定の定義
    lr: float = 1e-4
    ema_rate: str = "0.999,0.9999"  # comma-separated list of EMA values
    weight_decay: float = 1e-3
    lr_anneal_steps: int = 100000
    save_interval: int = 10000
    resume_checkpoint: str = ""
    use_bf16: bool = True
    drop_rate: float = 0.0
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    inference_scheduler: str = "ddim"
    p2_loss_k: float = 0.0
    p2_importance_sampling: bool = False


class ModelConfig(BaseModel):
    # 複数の設定に跨る値を定義
    image_size: int | None = None
    num_classes: int | None = None
    # 他設定の定義
    model_channels: int = 128
    num_res_blocks: int = 2
    num_heads: int = 1
    num_head_channels: int = 64
    num_heads_upsample: int = -1
    attention_resolutions: str = "32,16,8"
    channel_mult: str = ""
    dropout: float = 0.0
    use_checkpoint: bool = True
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    predict_sigma: bool = False  # Schedulerに従って自動で切替
    use_sdpa_attn: bool = False
    cond_spec: dict[str, dict[str, str | int]] = {
        "gender": {"type": "categorical", "num_classes": 2},
    }


class SchedulerConfig(BaseModel):
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"  # 他squaredcos_cap_v2等
    prediction_type: str = "epsilon"  # ε予測
    variance_type: str = "fixed_small"  # or fixed_small
    clip_sample: bool = True


class Config(BaseSettings):
    save_dir_root: str = "./results/"

    # 複数の設定に跨る値を定義
    image_size: int = Field(128)
    batch_size: int = Field(8)
    num_classes: int = Field(19)
    grayscale: bool = Field(False)

    train: TrainConfig = TrainConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",  # 不要なら削除
    )

    @model_validator(mode="after")
    def _update(self):
        self.model.predict_sigma = "learn" in self.scheduler.variance_type
        return self

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
