import copy
import os
from pathlib import Path
from typing import Iterator

import blobfile as bf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
from diffusers.schedulers import DDPMScheduler
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import vb_terms_bits_per_dim
from .model import UNetModel, update_ema

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        model: UNetModel,
        scheduler: DDPMScheduler,
        data: Iterator[DataLoader],
        num_classes: int,
        batch_size: int,
        lr: float,
        ema_rate: str,
        drop_rate: float,
        save_interval: int,
        resume_checkpoint: str,
        num_train_timesteps: int = 1000,
        use_fp16: bool = True,
        fp16_scale_growth: float = 1e-3,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        grayscale: bool = False,
    ):
        self.model: UNetModel = model
        self.scheduler: DDPMScheduler = scheduler
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.save_interval = save_interval
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint else None
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = UniformSampler(num_train_timesteps)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.grayscale = grayscale

        self.step = 1
        self.resume_step = 0

        self._load_checkpoint()

        self.scaler = GradScaler(enabled=self.use_fp16)
        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_models = [self._load_ema_model(rate) for rate in self.ema_rate]
            self.ema_models = [model for model in self.ema_models if model is not None]
        else:
            self.ema_models = [copy.deepcopy(self.model).eval() for _ in self.ema_rate]
            for m in self.ema_models:
                for p in m.parameters():
                    p.requires_grad_(False)

    def _load_checkpoint(self):
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            self.model.load_state_dict(th.load(self.resume_checkpoint))

    def _load_ema_model(self, rate):
        ema_checkpoint = find_ema_checkpoint(
            self.resume_checkpoint, self.resume_step, rate
        )
        if ema_checkpoint:
            model = copy.deepcopy(self.model)
            state_dict = th.load(ema_checkpoint)
            ema_model = model.load_state_dict(state_dict)
            return ema_model
        else:
            None

    def _load_optimizer_state(self):
        opt_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            self.opt.load_state_dict(th.load(opt_checkpoint))

            if self.opt.param_groups[0]["lr"] != self.lr:
                self.opt.param_groups[0]["lr"] = self.lr

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            cond = self.preprocess_input(cond)
            self.run_step(batch, cond)
            if self.step % self.save_interval == 0:
                self.save()
                self.sanity_test(batch=batch, device="cuda", cond=cond)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.sanity_test(batch=batch, device="cuda", cond=cond)

    def run_step(self, batch: torch.Tensor, cond: dict[str, torch.Tensor]):
        self.opt.zero_grad()
        batch = batch.to("cuda")
        cond = {k: v.to("cuda") for k, v in cond.items()}
        t, weights = self.schedule_sampler.sample(batch.shape[0], "cuda")
        noise = torch.randn_like(batch, device="cuda")
        x_t = self.scheduler.add_noise(batch, noise, t)

        # AMP
        autocast_dtype = torch.float16 if self.use_fp16 else torch.float32
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            model_out = self.model(x_t, t, y=cond["y"])  # 2C 出力は現状維持
            eps_pred, var_raw = torch.split(model_out, batch.shape[1], dim=1)

            # ====== MSE(ε) ======
            mse = ((noise - eps_pred) ** 2).mean(dim=(1, 2, 3))  # mean_flat と同等

            # ====== VB 項（KL or NLL@t=0） ======
            vb = vb_terms_bits_per_dim(
                scheduler=self.scheduler,
                x_start=batch,
                x_t=x_t,
                t=t,
                eps_pred_detached=eps_pred.detach(),  # meanには勾配を流さない
                var_raw=var_raw,
            )
            # total loss
            loss = ((mse + vb) * weights).mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self._update_ema()
        self._anneal_lr()

        print(f"\rstep: {self.step} | loss: {loss.detach().cpu().numpy():.4f}", end="")

    def _update_ema(self):
        # 既存の update_ema(target_params, source_params, rate) を再利用
        for rate, ema_model in zip(self.ema_rate, self.ema_models):
            update_ema(
                ema_model.parameters(), self.model.parameters(), rate=float(rate)
            )

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        step = f"{(self.step + self.resume_step):06d}"
        # main
        torch.save(self.model.state_dict(), f"./results/model{step}.pt")
        # EMA
        for rate, ema_model in zip(self.ema_rate, self.ema_models):
            torch.save(ema_model.state_dict(), f"./results/ema_{rate}_{step}.pt")
        # optimizer
        torch.save(self.opt.state_dict(), f"./results/opt{step}.pt")

    def sanity_test(
        self, batch: torch.Tensor, device: str, cond: dict[str, torch.Tensor]
    ):
        self.model.eval()
        x = torch.randn_like(batch, device=device)
        cond = {k: v.to(device) for k, v in cond.items()}
        total_steps = len(self.scheduler.timesteps)
        snapshot_names = ["25%", "50%", "75%"]
        snapshot_steps = [total_steps // 4 * i for i in range(1, 4)]
        snapshots = dict()
        with th.no_grad():
            total = len(self.scheduler.timesteps)
            for i, t in tqdm(enumerate(self.scheduler.timesteps), total=total):
                t_batch = torch.tensor([t] * x.shape[0], device=device)  # shape = [B]
                model_out = self.model(x, t_batch, y=cond["y"])  # 2C出力
                x_prev = self.scheduler.step(model_out, t, x).prev_sample
                # snapshot 用に x_prev を保存
                if i in snapshot_steps:
                    idx = snapshot_steps.index(i)
                    snapshots[snapshot_names[idx]] = (x_prev + 1) / 2.0
                x = x_prev
        self.model.train()

        src_img = ((batch + 1.0) / 2.0).to(device)
        inference_img = (x + 1) / 2.0
        log_images(
            inference_img=inference_img,
            src_img=src_img,
            snapshots=snapshots,
            output_dir="./results",
            step=self.step,
            grayscale=self.grayscale,
        )

    def preprocess_input(self, data: dict[str, torch.Tensor]):
        # move to GPU and change data types
        data["label"] = data["label"].long()

        # create one-hot label map
        label_map = data["label"]
        bs, _, h, w = label_map.size()
        nc = self.num_classes
        input_label = th.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        if self.drop_rate > 0.0:
            mask = (
                th.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate
            ).float()
            input_semantics = input_semantics * mask

        cond = {
            key: value
            for key, value in data.items()
            if key not in ["label", "path", "label_ori"]
        }
        cond["y"] = input_semantics

        return cond


class UniformSampler:
    def __init__(self, diffusion_steps: int):
        self._weights = np.ones([diffusion_steps])

    def weights(self):
        return self._weights

    def sample(self, batch_size: int, device: str):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


def parse_resume_step_from_filename(filename: Path | None):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    if filename is None:
        return 0
    if not filename.exists():
        return 0
    step = filename.stem.replace("model", "")
    try:
        return int(step)
    except ValueError:
        return 0


def find_ema_checkpoint(
    main_checkpoint: Path | None, step: int, rate: str
) -> Path | None:
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_images(
    inference_img: torch.Tensor,
    src_img: torch.Tensor,
    snapshots: dict[str, torch.Tensor],
    output_dir: str,
    step: int,
    grayscale: bool,
):
    num_rows = 2 + len(snapshots)
    num_cols = inference_img.shape[0]
    base_width = 4
    base_height = 4
    fig_width = num_cols * base_width + 2
    fig_height = num_rows * base_height

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.suptitle("Diffusion Model Results", fontsize=16)

    src_img = src_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
    inference_img = inference_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
    c = 1 if grayscale else 3
    src_img = np.clip(src_img[..., :c], 0, 1)
    inference_img = np.clip(inference_img[..., :c], 0, 1)

    kwargs = dict(cmap="gray") if grayscale else dict()
    for k in range(num_cols):
        axs[0, k].imshow(src_img[k], **kwargs)
        axs[0, k].axis("off")

        axs[1, k].imshow(inference_img[k], **kwargs)
        axs[1, k].axis("off")

        for i, snap in enumerate(snapshots.values()):
            tmp = np.clip(snap[k, :c, ...].cpu().detach().numpy(), 0, 1)
            axs[i + 2, k].imshow(tmp.transpose(1, 2, 0), **kwargs)
            axs[i + 2, k].axis("off")

    axs[0, 0].set_title("Source Image")
    axs[1, 0].set_title("Inference Image")
    for i, snap in enumerate(snapshots):
        axs[i + 2, 0].set_title(f"Snapshot {snap}")

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(f"{output_dir}/diffusion_results_{str(step).zfill(6)}.png")
    plt.close()
