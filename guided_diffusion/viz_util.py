from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def make_palette(nc: int, bg0_black: bool = True) -> np.ndarray:
    """クラス数 nc のラベルカラーパレットを作る (HUSLに近い均等色)。"""
    np.random.default_rng(0)
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


def log_images(
    src_img: torch.Tensor,
    cond_map: torch.Tensor,
    inference_img: torch.Tensor,
    snapshots: dict[str, torch.Tensor],
    output_dir: Path,
    class_num: int,
    step: int,
    grayscale: bool,
):
    """
    上段から:
      0: 元画像 (Source Image)
      1: ラベル (入力) をカラー化したもの (Label (input))
      2: 出力 (Inference Image)
      3~: スナップショット
    """
    output_dir.mkdir(exist_ok=True)

    # 画像を uint8 (B,H,W,3) に整形
    inference_img = to_uint8_image_batch(inference_img, grayscale)
    src_img = to_uint8_image_batch(src_img, grayscale)
    snapshots = {k: to_uint8_image_batch(i, grayscale) for k, i in snapshots.items()}

    assert cond_map.dim() == 4
    if cond_map.shape[1] > 1 and cond_map.dtype != torch.long:
        lab = cond_map.argmax(dim=1, keepdim=True)
    else:
        lab = cond_map.long()
    label_u8 = labels_to_color(lab, make_palette(class_num))

    # sigma予測してるかもしれないので
    c = 1 if grayscale else 3
    src_img = src_img[..., :c]
    inference_img = inference_img[..., :c]

    # 出力の形状を決定
    num_rows = 3 + len(snapshots)
    num_cols = inference_img.shape[0]
    base_width = 4
    base_height = 4
    fig_width = num_cols * base_width + 2
    fig_height = num_rows * base_height

    # 描画
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.suptitle("Diffusion Model Results", fontsize=16)

    kwargs = dict(cmap="gray") if grayscale else dict()
    for k in range(num_cols):
        axs[0, k].imshow(src_img[k], **kwargs)
        axs[0, k].axis("off")

        axs[1, k].imshow(label_u8[k], **kwargs)
        axs[1, k].axis("off")

        axs[2, k].imshow(inference_img[k], **kwargs)
        axs[2, k].axis("off")

        for i, snap in enumerate(snapshots.values()):
            axs[i + 3, k].imshow(snap[k, ..., :c], **kwargs)
            axs[i + 3, k].axis("off")

    axs[0, 0].set_title("Source Image")
    axs[1, 0].set_title("Input Label")
    axs[2, 0].set_title("Inference Result")
    for i, snap in enumerate(snapshots):
        axs[i + 3, 0].set_title(f"Snapshot {snap}")

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(output_dir / f"{str(step).zfill(6)}.png")
    plt.close()
