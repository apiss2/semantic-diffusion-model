import copy
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import vb_terms_bits_per_dim
from .model import UNetModel, update_ema
from .viz_util import log_images


def create_inference_scheduler(name: str, train_scheduler: DDPMScheduler):
    """推論サンプラーを学習スケジューラ設定から生成（beta/prediction_type等の整合を保証）"""
    name = (name or "ddim").lower()
    if name in ("ddim", "ddimscheduler"):
        cls = DDIMScheduler
    elif name in ("dpm", "dpmsolver", "dpm-solver", "dpm++", "dpmsolvermultistep"):
        cls = DPMSolverMultistepScheduler
    elif name in ("unipc", "unipc-multistep", "unipcmultistep"):
        cls = UniPCMultistepScheduler
    else:
        raise ValueError(f"unknown inference scheduler: {name}")
    return cls.from_config(train_scheduler.config)


class TrainLoop:
    def __init__(
        self,
        model: UNetModel,
        scheduler: DDPMScheduler,
        data: Iterator[DataLoader],
        save_dir: Path,
        num_classes: int,
        batch_size: int,
        lr: float,
        ema_rate: str,
        drop_rate: float,
        save_interval: int,
        resume_checkpoint: str,
        use_bf16: bool = True,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        grayscale: bool = False,
        grad_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        inference_scheduler: str = "ddim",
        p2_loss_k: float = 0.0,
        p2_importance_sampling: bool = False,
    ):
        self.model: UNetModel = model
        self.scheduler: DDPMScheduler = scheduler
        self.data = data
        self.save_dir = save_dir
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
        self.resume_checkpoint = (
            Path(resume_checkpoint) if resume_checkpoint != "" else None
        )
        self.use_bf16 = use_bf16
        self.schedule_sampler = UniformSampler(
            scheduler=self.scheduler,
            p2_loss_k=p2_loss_k,
            importance_sampling=p2_importance_sampling,
        )
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.grayscale = grayscale

        mixed_precision = "bf16" if use_bf16 else "no"
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accumulation_steps,
        )
        self.device = self.accelerator.device
        self.max_grad_norm = max_grad_norm
        self.inference_scheduler_default = inference_scheduler

        self.step = 1
        self.resume_step = 0

        # --- 既存のチェックポイント読み込み（prepare 前）---
        self._load_checkpoint()

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # --- DDP wrap ---
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if self.resume_step:
            self._load_optimizer_state()
            self.ema_models = [self._load_ema_model(rate) for rate in self.ema_rate]
            self.ema_models = [m for m in self.ema_models if m is not None]
        else:
            self.ema_models = [
                copy.deepcopy(self.accelerator.unwrap_model(self.model)).eval()
                for _ in self.ema_rate
            ]
            for m in self.ema_models:
                for p in m.parameters():
                    p.requires_grad_(False)
        # TensorBoard（rank0 のみ）
        self.tb: SummaryWriter | None = None
        if self.accelerator.is_main_process:
            self.tb = SummaryWriter(log_dir=str(self.save_dir / "tb"))

    def _load_checkpoint(self):
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            state = torch.load(self.resume_checkpoint, map_location="cpu")
            self.model.load_state_dict(state)

    def _load_ema_model(self, rate):
        ema_checkpoint = find_ema_checkpoint(
            self.resume_checkpoint, self.resume_step, rate
        )
        if ema_checkpoint:
            model = copy.deepcopy(self.model).to("cpu")
            state_dict = torch.load(ema_checkpoint, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            return model.eval()
        else:
            return None

    def _load_optimizer_state(self):
        if self.resume_checkpoint:
            name = f"ckpt/opt{self.resume_step:06}.pt"
            opt_checkpoint = self.save_dir / name
            assert opt_checkpoint.exists()
            opt_state = torch.load(opt_checkpoint, map_location="cpu")
            self.opt.load_state_dict(opt_state)
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
            self.step += 1
        # 最終保存
        last = (self.step - 1) % self.save_interval != 0
        if last and self.accelerator.is_main_process:
            self.save()
            self.sanity_test(batch=batch, device=str(self.device), cond=cond)

    def run_step(self, batch: torch.Tensor, cond: dict[str, torch.Tensor]):
        self.opt.zero_grad()
        batch = batch.to("cuda")
        semantic_map = cond["semantic_map"].to("cuda")
        if "conds" in cond:
            conds = {k: v.long().to("cuda") for k, v in cond["conds"].items()}
        t, weights = self.schedule_sampler.sample(batch.shape[0], "cuda")
        noise = torch.randn_like(batch, device="cuda")
        x_t = self.scheduler.add_noise(batch, noise, t)

        self.opt.zero_grad(set_to_none=True)

        # AMP
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                _cond = {"semantic_map": semantic_map}
                if "conds" in cond:
                    _cond["conds"] = conds
                outputs = self.model(sample=x_t, timestep=t, added_cond_kwargs=_cond)
                model_out = outputs.sample
                if self.model.predict_sigma:
                    eps_pred, var_raw = torch.chunk(model_out, 2, dim=1)
                    # ====== MSE(ε) ======
                    loss = ((noise - eps_pred) ** 2).mean(dim=(1, 2, 3))
                    # ====== VB 項（KL or NLL@t=0） ======
                    loss += vb_terms_bits_per_dim(
                        scheduler=self.scheduler,
                        x_start=batch,
                        x_t=x_t,
                        t=t,
                        eps_pred_detached=eps_pred.detach(),  # meanには勾配を流さない
                        var_raw=var_raw,
                    )
                else:
                    loss = ((noise - model_out) ** 2).mean(dim=(1, 2, 3))
                # total loss
                loss = (loss * weights).mean()

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients and self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            # grad norm（記録用）
            grad_norm = None
            if self.accelerator.sync_gradients:
                # clip と同じタイミングで OK（クリップ後の値を記録）
                if self.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                # 総合ノルムを計算
                total_sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        total_sq += float(g.norm(2).item() ** 2)
                grad_norm = float(total_sq**0.5)

            if self.accelerator.sync_gradients:
                self.opt.step()

        self._update_ema()
        self._anneal_lr()

        _loss = loss.detach().float().cpu().item()
        self.accelerator.print(f"\rstep: {self.step} | loss: {loss:.4f}", end="")

        # ===== TensorBoard =====
        if self.tb is not None:
            gstep = self.step + self.resume_step
            self.tb.add_scalar("train/loss", _loss, gstep)
            self.tb.add_scalar("train/lr", self.opt.param_groups[0]["lr"], gstep)
            if grad_norm is not None:
                self.tb.add_scalar("train/grad_norm", grad_norm, gstep)
            # ema decay（定数でも“随時”出す）
            for i, r in enumerate(self.ema_rate):
                self.tb.add_scalar(f"ema/decay_{i}", float(r), gstep)
            # 100step ごとに EMA と本体の L2 距離も出す（任意）
            if (gstep % 100 == 0) and (len(self.ema_models) > 0):
                ema0 = self.ema_models[0]
                dist_sq = 0.0
                for p, q in zip(
                    self.accelerator.unwrap_model(self.model).parameters(),
                    ema0.parameters(),
                ):
                    dist_sq += float((p.detach() - q.detach()).pow(2).sum().item())
                self.tb.add_scalar("ema/distance_l2", dist_sq**0.5, gstep)

    def _update_ema(self):
        src_params = self.accelerator.unwrap_model(self.model).parameters()
        for rate, ema_model in zip(self.ema_rate, self.ema_models):
            update_ema(ema_model.parameters(), src_params, rate=float(rate))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def save(self):
        step = f"{(self.step + self.resume_step):06d}"
        sd = self.save_dir / "ckpt"
        sd.mkdir(exist_ok=True)

        # unwrap した state_dict を保存（DDP wrapperを含めない）
        state = self.accelerator.get_state_dict(self.model)
        torch.save(state, sd / f"model{step}.pt")

        # EMA
        for rate, ema_model in zip(self.ema_rate, self.ema_models):
            torch.save(ema_model.state_dict(), sd / f"ema_{rate}_{step}.pt")

        # optimizer（rank0 のみ）
        if self.accelerator.is_main_process:
            torch.save(self.opt.state_dict(), sd / f"opt{step}.pt")

    def sanity_test(
        self, batch: torch.Tensor, device: str, cond: dict[str, torch.Tensor]
    ):
        self.accelerator.unwrap_model(self.model).eval()
        x = torch.randn_like(batch, device=device) * self.scheduler.init_noise_sigma
        cond["semantic_map"] = cond["semantic_map"].to(device)
        if "conds" in cond:
            cond["conds"] = {k: v.to(device) for k, v in cond["conds"].items()}
        total_steps = len(self.scheduler.timesteps)
        snapshot_names = ["25%", "50%", "75%"]
        snapshot_steps = [total_steps // 4 * i for i in range(1, 4)]
        snapshots = dict()
        with torch.no_grad():
            _iter = enumerate(self.scheduler.timesteps)
            disable = not self.accelerator.is_main_process
            for i, t in tqdm(_iter, total=total_steps, disable=disable):
                _t = torch.tensor([t] * x.shape[0], device=device)
                outputs = self.accelerator.unwrap_model(self.model)(
                    sample=x, timestep=_t, added_cond_kwargs=cond
                )
                x_prev = self.scheduler.step(outputs.sample, t, x).prev_sample
                x_prev = x_prev.clamp(-1, 1)
                if i in snapshot_steps:
                    idx = snapshot_steps.index(i)
                    snapshots[snapshot_names[idx]] = x_prev
                x = x_prev
        self.accelerator.unwrap_model(self.model).train()

        if self.accelerator.is_main_process:
            log_images(
                src_img=batch.to(device),
                cond_map=cond["semantic_map"],
                inference_img=x,
                snapshots=snapshots,
                output_dir=self.save_dir / "snapshots",
                class_num=self.num_classes,
                step=self.step,
                grayscale=self.grayscale,
            )

    @torch.no_grad()
    def generate_with_cfg(
        self,
        batch_size: int,
        cond: dict[str, torch.Tensor],
        *,
        scheduler_name: str | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        use_ema: bool = True,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        CFG (eps = eps_u + w*(eps_c - eps_u)) による生成。
        条件は cond["semantic_map"] を使用。uncond は all-zero の one-hot を用いる。
        """
        device = self.device
        model_for_infer = (
            self.ema_models[0]
            if (use_ema and len(self.ema_models) > 0)
            else self.accelerator.unwrap_model(self.model)
        )
        model_for_infer = model_for_infer.to(device).eval()

        # 推論サンプラーを切替
        name = scheduler_name or self.inference_scheduler_default
        sched = create_inference_scheduler(name, self.scheduler)
        sched.set_timesteps(num_inference_steps, device=device)

        x = torch.randn(
            batch_size,
            model_for_infer.in_channels,
            self.accelerator.unwrap_model(self.model).image_size,
            self.accelerator.unwrap_model(self.model).image_size,
            device=device,
        )

        # cond/uncond の準備
        cond_map = cond["semantic_map"].to(device)
        if cond_map.shape[0] != batch_size:
            # 最低限の合わせ込み
            cond_map = cond_map[:batch_size]
        uncond_map = torch.zeros_like(cond_map)

        snapshots = {}
        snap_points = {
            int(num_inference_steps * 0.25): "25%",
            int(num_inference_steps * 0.5): "50%",
            int(num_inference_steps * 0.75): "75%",
        }

        for i, t in enumerate(sched.timesteps):
            # diffusers の API に合わせる：t は shape=[B] でもスカラでも可
            t_b = t

            # two-pass
            eps_c = model_for_infer(
                sample=x, timestep=t_b, added_cond_kwargs={"semantic_map": cond_map}
            ).sample
            if model_for_infer.predict_sigma:
                eps_c, var_c = torch.chunk(eps_c, 2, dim=1)

            eps_u = model_for_infer(
                sample=x, timestep=t_b, added_cond_kwargs={"semantic_map": uncond_map}
            ).sample
            if model_for_infer.predict_sigma:
                eps_u, _ = torch.chunk(eps_u, 2, dim=1)

            eps = eps_u + guidance_scale * (eps_c - eps_u)

            if model_for_infer.predict_sigma:
                model_out = torch.cat(
                    [eps, var_c], dim=1
                )  # var は cond 側を使用（一般的な実装）
            else:
                model_out = eps

            x = sched.step(model_out, t, x).prev_sample

            if return_intermediate and i in snap_points:
                snapshots[snap_points[i]] = (x + 1) / 2.0

        out = (x + 1) / 2.0
        return (out, snapshots) if return_intermediate else out

    def preprocess_input(self, data: dict[str, torch.Tensor]):
        # move to GPU and change data types
        data["label"] = data["label"].long()

        # create one-hot label map
        label_map = data["label"]
        bs, _, h, w = label_map.size()
        nc = self.num_classes
        device = label_map.device
        input_label = torch.FloatTensor(bs, nc, h, w, device=device).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        if self.drop_rate > 0.0:
            mask = (
                torch.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate
            ).float()
            input_semantics = input_semantics * mask

        cond = {
            key: value for key, value in data.items() if key not in ["label", "path"]
        }
        cond["semantic_map"] = input_semantics
        return cond


class UniformSampler:
    def __init__(
        self,
        scheduler: DDPMScheduler,
        p2_loss_k: float = 0.0,
        importance_sampling: bool = False,
    ):
        self.importance_sampling = importance_sampling
        # ᾱ_t（DDPM の cumulative product）に基づく p2 係数
        alphas_cumprod = scheduler.alphas_cumprod.detach().cpu().numpy()
        if p2_loss_k > 0.0:
            p2 = (1.0 - alphas_cumprod) ** float(p2_loss_k)
        else:
            p2 = np.ones_like(alphas_cumprod)
        # サンプリング分布
        self._weights = p2.copy() if importance_sampling else np.ones_like(p2)
        # ロスに掛ける係数（IS しないなら (1-ᾱ)^p をここで使う）
        self._p2_loss_weights = p2

    def weights(self):
        return self._weights

    def sample(self, batch_size: int, device: str):
        """
        Importance-sample timesteps for a batch.
        return: (timesteps, weights)
        weights は unbiased 化のための 1/(N p[t]) に p2 loss 係数を掛けたもの。
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights_np = weights_np * self._p2_loss_weights[indices_np]
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
    path = main_checkpoint.parent.joinpath(filename)
    if path.exists():
        return path
    return None
