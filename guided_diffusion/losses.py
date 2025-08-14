"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch
from diffusers.schedulers import DDPMScheduler


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def _gather(arr, t, shape, device):
    # numpy/torch配列を t に合わせて broadcast 抽出
    vals = torch.as_tensor(arr, dtype=torch.float32, device=device)
    out = vals.gather(0, t).view(-1, *([1] * (len(shape) - 1)))
    return out


def q_posterior_stats(betas, ac, x0, xt, t):
    # guided-diffusionの式をschedulerの係数で
    ac_prev = torch.cat([torch.ones_like(ac[:1]), ac[:-1]], dim=0)

    var = betas * (1 - ac_prev) / (1 - ac)
    logv = torch.log(torch.cat([var[1:2], var[1:]], dim=0))  # clipped 同等に要調整

    c1 = betas * torch.sqrt(ac_prev) / (1 - ac)
    c2 = (1 - ac_prev) * torch.sqrt(1 - betas) / (1 - ac)
    c1 = _gather(c1, t, xt.shape, xt.device)
    c2 = _gather(c2, t, xt.shape, xt.device)

    mean = c1 * x0 + c2 * xt
    var_t = _gather(var, t, xt.shape, xt.device)
    logv_t = _gather(logv, t, xt.shape, xt.device)
    return mean, var_t, logv_t


def predict_x0_from_eps(ac, xt, t, eps):
    sr = _gather(torch.sqrt(1.0 / ac), t, xt.shape, xt.device)
    srm = _gather(torch.sqrt(1.0 / ac - 1.0), t, xt.shape, xt.device)
    return sr * xt - srm * eps


def vb_terms_bits_per_dim(
    scheduler: DDPMScheduler, x_start, x_t, t, eps_pred_detached, var_raw
):
    # learned_range: raw∈[-1,1] → logv = frac*max + (1-frac)*min
    betas = scheduler.betas.to("cuda")
    ac = scheduler.alphas_cumprod.to("cuda")
    post_var = betas * (1 - torch.cat([ac[:1].new_ones(1), ac[:-1]], 0)) / (1 - ac)
    min_log = _gather(
        torch.log(torch.cat([post_var[1:2], post_var[1:]], 0)), t, x_t.shape, x_t.device
    )
    max_log = _gather(torch.log(betas), t, x_t.shape, x_t.device)
    frac = (var_raw + 1) / 2
    model_logv = frac * max_log + (1 - frac) * min_log

    # meanは x0(pred) を介して q_post のmeanを使用（勾配は流さない想定）
    x0_pred = predict_x0_from_eps(ac, x_t, t, eps_pred_detached)
    model_mean, _, _ = q_posterior_stats(betas, ac, x0_pred, x_t, t)

    true_mean, _, true_logv = q_posterior_stats(betas, ac, x_start, x_t, t)
    kl = normal_kl(true_mean, true_logv, model_mean, model_logv).mean(
        dim=(1, 2, 3)
    ) / np.log(2.0)

    nll = -discretized_gaussian_log_likelihood(
        x_start, means=model_mean, log_scales=0.5 * model_logv
    ).mean(dim=(1, 2, 3)) / np.log(2.0)

    return torch.where(t == 0, nll, kl)  # t=0はNLL、それ以外はKL
