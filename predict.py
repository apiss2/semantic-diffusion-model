# predict.py
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# diffusers schedulers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# プロジェクト内モジュール
from config import Config
from guided_diffusion.image_datasets import SDMDataset
from guided_diffusion.model import UNetModel


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="学習済み重みへのパス (…/ckpt/modelXXXXXX.pt や ema_*.pt)",
    )
    p.add_argument("--outdir", type=str, default="./inference", help="推論結果の保存先")
    p.add_argument(
        "--num_images",
        type=int,
        default=8,
        help="生成するサンプル数（=推論に使うバッチサイズ）",
    )
    p.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="サンプリング反復回数 (DDIM/DDPM 共通)",
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="サンプラー種別",
    )
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0=deterministic)")
    p.add_argument(
        "--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic_loader",
        action="store_true",
        help="データローダのシャッフルを無効化",
    )
    args = p.parse_args()
    return args


# ----------------------------
# メイン
# ----------------------------
def main():
    assert torch.cuda.is_available(), "CUDAが必要です"
    device = torch.device("cuda")
    args = get_args()
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt).resolve()
    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"

    # save_dir を推定し、config.json をロード（学習時の設定を復元） :contentReference[oaicite:6]{index=6}
    save_dir = ckpt_path.parent.parent
    cfg_path = save_dir / "config.json"
    if cfg_path.exists():
        cfg_dict = json.loads(cfg_path.read_text())
        config = Config.model_validate(cfg_dict)
    else:
        # フォールバック: 現在の環境からConfigを組む
        config = Config()

    out_root = Path(args.outdir)
    ts_name = datetime.now().strftime("%Y-%m%d-%H%M%S")
    run_dir = out_root / ts_name
    (run_dir / "grids").mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    # ---------------- モデル & スケジューラ ----------------
    model = UNetModel(**config.model.model_dump()).to(device).eval()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    if args.sampler == "ddim":
        scheduler = DDIMScheduler(
            beta_schedule=config.scheduler.beta_schedule,
            prediction_type=config.scheduler.prediction_type,
            clip_sample=config.scheduler.clip_sample,
            num_train_timesteps=config.num_train_timesteps,
        )
        scheduler.set_timesteps(args.num_steps, device=device)
        scheduler.eta = args.eta  # η
    else:
        scheduler = DDPMScheduler(
            beta_schedule=config.scheduler.beta_schedule,
            prediction_type=config.scheduler.prediction_type,
            variance_type=config.scheduler.variance_type,  # learned_range を保持
            clip_sample=config.scheduler.clip_sample,
            num_train_timesteps=config.num_train_timesteps,
        )
        scheduler.set_timesteps(args.num_steps, device=device)

    # ---------------- データセット ----------------
    # SDMDataset は __getitem__ で (img, {"path":..., "label":(1,H,W)}) を返す
    ds = SDMDataset(
        image_dir=config.dataset.image_dir,
        label_dir=config.dataset.label_dir,
        num_classes=config.num_classes,
        size=config.image_size,
        grayscale=config.grayscale,
    )
    # 推論は「見本として」num_images 分のみ読む（シャッフル可）
    if args.deterministic_loader:
        indices = np.arange(min(args.num_images, len(ds)))
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(ds), size=min(args.num_images, len(ds)), replace=False)

    # 1バッチ推論：可視化の都合でまとめて処理
    imgs = []
    labels = []
    names = []
    for i in indices:
        img, cond = ds[i]
        imgs.append(img)
        labels.append(torch.from_numpy(cond["label"]))
        # 元ファイル名を出力名に使用
        names.append(Path(cond["path"]).stem)
    batch = torch.stack(imgs, dim=0).to(device)  # (B,C,H,W) [-1,1]
    labels_b1hw = torch.stack(labels, dim=0).to(device)  # (B,1,H,W)

    # ---------------- サンプリング ----------------
    # 入力画像は可視化にのみ使用（sanity_testと同様） :contentReference[oaicite:10]{index=10}
    use_amp = {"fp32": False, "fp16": True, "bf16": True}[args.precision]
    gen_xT, snapshots = sample_images(
        model=model,
        scheduler=scheduler,
        labels_b1hw=labels_b1hw,
        shape_chw=(batch.shape[1], batch.shape[2], batch.shape[3]),
        device=device,
        use_amp=use_amp,
    )

    # ---------------- 保存 ----------------
    # グリッド: Source / Label / Generated / Snapshots(25/50/75%)
    palette = make_palette(config.num_classes)
    save_grid_with_snapshots(
        outdir=run_dir / "grids",
        step_name="grid",
        src_bchw=batch,
        gen_bchw=gen_xT,
        labels_b1hw=labels_b1hw,
        snapshots=snapshots,
        grayscale=config.grayscale,
        palette=palette,
    )

    # 個別保存（生成結果のみ）
    save_each_image(run_dir / "images", names, gen_xT, grayscale=config.grayscale)

    print("\nSaved:")
    print(f"  Grid : {run_dir / 'grids' / 'grid.png'}")
    print(f"  Images -> {run_dir / 'images'}")


# ----------------------------
# 可視化ユーティリティ
# ----------------------------
def make_palette(nc: int, bg0_black: bool = True) -> np.ndarray:
    """クラス数 nc のラベルカラーパレットを作る (HUSLに近い均等色)。"""
    rng = np.random.default_rng(0)
    # 均等分布の色相をベースに、固定シードで安定化
    hues = np.linspace(0, 1, nc, endpoint=False)
    sat, val = 0.75, 1.0
    palette = []
    for i, h in enumerate(hues):
        if bg0_black and i == 0:
            palette.append((0, 0, 0))
            continue
        # HSV -> RGB
        r, g, b = hsv_to_rgb(h, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(palette, dtype=np.uint8)


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def labels_to_color(label_tensor: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """
    (B,1,H,W) の int ラベルを (B,H,W,3) のRGB画像にする。
    """
    lab = label_tensor.squeeze(1).cpu().numpy().astype(np.int64)  # (B,H,W)
    colored = palette[lab]  # (B,H,W,3)
    return colored


def to_uint8_image_batch(t: torch.Tensor, grayscale: bool) -> np.ndarray:
    """
    [-1,1] の (B,C,H,W) を (B,H,W,C) uint8 [0,255] に。
    """
    x = ((t.clamp(-1, 1) + 1.0) * 0.5).detach().cpu().numpy()
    if grayscale:
        x = x[:, :1, ...].transpose(0, 2, 3, 1)  # (B,H,W,1)
        x = np.repeat(x, 3, axis=3)  # 3ch化して保存互換に
    else:
        x = x[:, :3, ...].transpose(0, 2, 3, 1)  # (B,H,W,3)
    x = (x * 255.0).round().astype(np.uint8)
    return x


def save_grid_with_snapshots(
    outdir: Path,
    step_name: str,
    src_bchw: torch.Tensor,
    gen_bchw: torch.Tensor,
    labels_b1hw: torch.Tensor,
    snapshots: Dict[str, torch.Tensor],
    grayscale: bool,
    palette: np.ndarray,
):
    """
    行: [Source, Label, Generated, Snapshot 25%, 50%, 75%]
    列: バッチ内の各サンプル
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 画像化
    src_np = to_uint8_image_batch(src_bchw, grayscale)  # (B,H,W,3)
    gen_np = to_uint8_image_batch(gen_bchw, grayscale)  # (B,H,W,3)
    lab_np = labels_to_color(labels_b1hw, palette)  # (B,H,W,3)
    snap_nps = [
        to_uint8_image_batch(v, grayscale) for _, v in snapshots.items()
    ]  # list of (B,H,W,3)

    B, H, W, _ = src_np.shape
    rows = 3 + len(snap_nps)
    pad = 4  # 余白
    gap = 8  # 画像間の間隔
    grid_w = B * W + (B - 1) * gap + 2 * pad
    grid_h = rows * H + (rows - 1) * gap + 2 * pad
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    def paste_row(row_idx: int, imgs_bhwc: np.ndarray):
        y0 = pad + row_idx * (H + gap)
        for k in range(B):
            x0 = pad + k * (W + gap)
            canvas[y0 : y0 + H, x0 : x0 + W] = imgs_bhwc[k]

    paste_row(0, src_np)
    paste_row(1, lab_np)
    paste_row(2, gen_np)
    for i, snap in enumerate(snap_nps):
        paste_row(3 + i, snap)

    Image.fromarray(canvas).save(outdir / f"{step_name}.png")


def save_each_image(
    outdir: Path, names: list[str], gen_bchw: torch.Tensor, grayscale: bool
):
    outdir.mkdir(parents=True, exist_ok=True)
    gen_np = to_uint8_image_batch(gen_bchw, grayscale)
    for k, name in enumerate(names):
        Image.fromarray(gen_np[k]).save(outdir / f"{name}.png")


# ----------------------------
# ラベル→one-hot の前処理（TrainLoop.preprocess_input 相当）
# ----------------------------
def labels_to_onehot(
    labels_b1hw: torch.Tensor, num_classes: int, drop_rate: float = 0.0
) -> torch.Tensor:
    """
    (B,1,H,W) int -> (B,C,H,W) float
    """
    bs, _, h, w = labels_b1hw.size()
    input_label = torch.zeros(
        (bs, num_classes, h, w), device=labels_b1hw.device, dtype=torch.float32
    )
    input_semantics = input_label.scatter_(1, labels_b1hw.long(), 1.0)
    if drop_rate > 0.0:
        mask = (
            torch.rand([bs, 1, 1, 1], device=labels_b1hw.device) > drop_rate
        ).float()
        input_semantics = input_semantics * mask
    return input_semantics


# ----------------------------
# サンプリング本体
# ----------------------------
@torch.no_grad()
def sample_images(
    model: UNetModel,
    scheduler,
    labels_b1hw: torch.Tensor,
    shape_chw: tuple[int, int, int],
    device: torch.device,
    snapshots_at: tuple[float, float, float] = (0.25, 0.5, 0.75),
    use_amp: bool = True,
):
    """
    x_T ~ N(0,I) から条件付き生成。
    DDPM: model_output=[eps, var_raw] をそのまま渡す
    DDIM: model_output から eps のみ取り出して渡す
    """
    B = labels_b1hw.shape[0]
    C, H, W = shape_chw
    x = torch.randn((B, C, H, W), device=device)
    y = labels_to_onehot(
        labels_b1hw.to(device), num_classes=model.num_classes, drop_rate=0.0
    )

    total_steps = len(scheduler.timesteps)
    snap_indices = [int(total_steps * r) - 1 for r in snapshots_at]
    snap_names = [f"{int(r * 100)}%" for r in snapshots_at]
    snapshots: Dict[str, torch.Tensor] = {}

    # AMP
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    is_ddim = isinstance(scheduler, DDIMScheduler)

    for i, t in tqdm(
        enumerate(scheduler.timesteps), total=total_steps, desc="Sampling"
    ):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            model_out = model(x, t_batch, y=y)

            if is_ddim:
                # DDIM は ε のみ使用（分散は使わない）
                eps_pred, _ = torch.split(model_out, C, dim=1)
                step_out = scheduler.step(model_output=eps_pred, timestep=t, sample=x)
            else:
                # DDPM(learned_range) は [eps, var_raw] を渡せる
                step_out = scheduler.step(model_output=model_out, timestep=t, sample=x)

        x = step_out.prev_sample

        if i in snap_indices:
            name = snap_names[snap_indices.index(i)]
            snapshots[name] = (x + 1.0) / 2.0  # 可視化用に [0,1]

    return x, snapshots


if __name__ == "__main__":
    main()
