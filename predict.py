from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers.schedulers import DDPMScheduler

from config import Config
from guided_diffusion.model import UNetModel
from guided_diffusion.train_util import create_inference_scheduler
from argparse import ArgumentParser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--conds_json", type=Path, default=None)
    parser.add_argument("--scheduler", type=str, default=None, help="ddim|dpm|unipc")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def load_labels_as_onehot(
    label_paths: List[Path],
    image_size: int,
    num_classes: int,
) -> torch.Tensor:
    """
    各ラベル画像を (1,H,W)long -> (C=K,H,W) onehot(float) へ。
    すべて image_size にリサイズ (最近傍)。
    return: (B,K,H,W) float32
    """
    maps = []
    for p in label_paths:
        lab = Image.open(p)
        lab = lab.resize((image_size, image_size), Image.NEAREST)
        lab = np.array(lab, dtype=np.int64)
        lab = np.clip(lab, 0, num_classes - 1)  # クランプ
        lab = torch.from_numpy(lab[None, ...])  # (1,H,W)
        maps.append(lab)
    label = torch.stack(maps, dim=0)  # (B,1,H,W)
    b, _, h, w = label.shape
    oh = torch.zeros((b, num_classes, h, w), dtype=torch.float32)
    oh.scatter_(1, label, 1.0)
    return oh


@torch.no_grad()
def sample_with_cfg(
    model: UNetModel,
    train_scheduler: DDPMScheduler,
    cond_map: torch.Tensor,
    *,
    scheduler_name: str = "ddim",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
) -> torch.Tensor:
    """
    TrainLoop.generate_with_cfg() と同等の二値CFGサンプリング。
    - cond_map: (B,K,H,W) float（one-hot）
    return: (B,C,H,W) in [0,1]
    """
    model = model.eval().to(DEVICE)

    sched = create_inference_scheduler(scheduler_name, train_scheduler)
    sched.set_timesteps(num_inference_steps, device=DEVICE)

    B, K, H, W = cond_map.shape
    x = torch.randn(B, model.in_channels, H, W, device=DEVICE)

    uncond = torch.zeros_like(cond_map)

    for t in sched.timesteps:
        t_b = t  # diffusers の API はスカラ / ベクタ両対応

        eps_c = model(
            sample=x, timestep=t_b, added_cond_kwargs={"semantic_map": cond_map}
        ).sample
        if model.predict_sigma:
            eps_c, var_c = torch.chunk(eps_c, 2, dim=1)

        eps_u = model(
            sample=x, timestep=t_b, added_cond_kwargs={"semantic_map": uncond}
        ).sample
        if model.predict_sigma:
            eps_u, _ = torch.chunk(eps_u, 2, dim=1)

        eps = eps_u + guidance_scale * (eps_c - eps_u)
        model_out = (
            torch.cat([eps, var_c], dim=1)
            if getattr(model, "predict_sigma", False)
            else eps
        )

        x = sched.step(model_out, t, x).prev_sample

    return (x + 1) / 2.0  # [0,1]


def save_batch_images(
    x01: torch.Tensor, paths: List[Path], outdir: Path, grayscale: bool
):
    """
    x01: (B,C,H,W) in [0,1]
    """
    x = x01.clamp(0, 1).detach().cpu().numpy()
    if grayscale:
        x = x[:, :1, ...].transpose(0, 2, 3, 1)  # (B,H,W,1)
        x = np.repeat(x, 3, axis=3)
    else:
        x = x[:, :3, ...].transpose(0, 2, 3, 1)  # (B,H,W,3)

    x = (x * 255.0).round().astype(np.uint8)
    outdir.mkdir(parents=True, exist_ok=True)
    for arr, src in zip(x, paths):
        Image.fromarray(arr).save(outdir / f"{src.stem}.png")


def main():
    args = get_args()

    # ===== 設定の読み込み =====
    cfg = Config()  # .env も反映される
    image_size = cfg.image_size
    num_classes = cfg.num_classes
    grayscale = cfg.grayscale

    # 学習時スケジューラ設定を復元（beta/prediction_type/variance_type の整合）
    train_sched = DDPMScheduler(**cfg.scheduler.model_dump())

    # モデル構築 & 重み読み込み
    model = UNetModel(**cfg.model.model_dump()).to(DEVICE)
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print("[warn] missing keys:", missing)
    if len(unexpected) > 0:
        print("[warn] unexpected keys:", unexpected)

    # ===== 入力ラベルの収集 =====
    # 拡張子はよくあるものを一通り拾う
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    label_paths = []
    for e in exts:
        label_paths.extend(sorted(args.labels.glob(e)))
    if len(label_paths) == 0:
        raise FileNotFoundError(f"no label images found in: {args.labels}")

    # ===== バッチ処理 =====
    bs = max(1, int(args.batch))
    scheduler_name = (args.scheduler or cfg.train.inference_scheduler).lower()
    for i in tqdm(range(0, len(label_paths), bs), desc="predict"):
        chunk = label_paths[i : i + bs]
        # one-hot semantic map
        cond_map = load_labels_as_onehot(chunk, image_size, num_classes).to(DEVICE)

        # 生成
        x01 = sample_with_cfg(
            model,
            train_sched,
            cond_map,
            scheduler_name=scheduler_name,
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.scale),
        )

        # 保存
        save_batch_images(x01, chunk, args.outdir, grayscale)

    print(f"done. => {args.outdir}")


if __name__ == "__main__":
    main()
