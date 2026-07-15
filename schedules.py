"""
Time schedules for conditional probability paths and diffusion.

Provides α_t and β_t schedules for interpolating between distributions,
plus noise schedules (variance, SNR) for diffusion models (DDPM/DDIM).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from torch.func import jacrev, vmap

log = logging.getLogger(__name__)


class Alpha(ABC):
    """
    Alpha schedule: α: [0,1] → [0,1] with α_0 = 0, α_1 = 1
    Controls weight on target in interpolation.
    """
    
    def __init__(self):
        # Verify boundary conditions
        if not torch.allclose(self(torch.zeros(1)), torch.zeros(1)):
            raise ValueError("Alpha schedule must satisfy alpha(0) = 0")
        if not torch.allclose(self(torch.ones(1)), torch.ones(1)):
            raise ValueError("Alpha schedule must satisfy alpha(1) = 1")
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute α_t."""
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Compute dα_t/dt using autodiff."""
        t_in = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t_in)
        return dt.view_as(t)


class Beta(ABC):
    """
    Beta schedule: β: [0,1] → [0,1] with β_0 = 1, β_1 = 0
    Controls weight on source (noise) in interpolation.
    """
    
    def __init__(self):
        # Verify boundary conditions
        if not torch.allclose(self(torch.zeros(1)), torch.ones(1)):
            raise ValueError("Beta schedule must satisfy beta(0) = 1")
        if not torch.allclose(self(torch.ones(1)), torch.zeros(1)):
            raise ValueError("Beta schedule must satisfy beta(1) = 0")
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute β_t."""
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Compute dβ_t/dt using autodiff."""
        t_in = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t_in)
        return dt.view_as(t)


class LinearAlpha(Alpha):
    """Linear schedule: α_t = t"""
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class LinearBeta(Beta):
    """Linear schedule: β_t = 1 - t"""
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


class SquareRootBeta(Beta):
    """Square root schedule: β_t = √(1 - t)"""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 / torch.sqrt(1.0 - t)


# ---------------------------------------------------------------------------
# Diffusion noise schedules (DDPM / DDIM)
# ---------------------------------------------------------------------------
# These parameterise q(x_t | x_0) = N(x_t; sqrt_alpha_bar_t * x_0, (1 - alpha_bar_t) * I)
# using discrete or continuous-time formulations.
# ---------------------------------------------------------------------------

class DiffusionNoiseSchedule(ABC):
    """
    Abstract noise schedule for diffusion models.

    Defines ᾱ_t (alpha_bar), the cumulative noise level at each timestep,
    such that:
        q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1 − ᾱ_t) · I)

    Subclasses must implement ``alpha_bar(t)`` over t ∈ [0, 1] (continuous)
    or index t ∈ {0, …, T-1} (discrete).
    """

    @abstractmethod
    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative noise level ᾱ_t.  Same shape as *t*."""
        pass

    def sqrt_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """√ᾱ_t — signal scaling coefficient."""
        return torch.sqrt(self.alpha_bar(t))

    def sqrt_one_minus_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """√(1 − ᾱ_t) — noise scaling coefficient."""
        return torch.sqrt(1.0 - self.alpha_bar(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio: ᾱ_t / (1 − ᾱ_t)."""
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab + 1e-8)

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """log SNR = log(ᾱ_t) − log(1 − ᾱ_t)."""
        ab = self.alpha_bar(t).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(ab) - torch.log(1.0 - ab)


class LinearDiffusionSchedule(DiffusionNoiseSchedule):
    """
    Continuous variance-preserving linear β schedule.

    β(t) = β_min + (β_max − β_min) · t
    ᾱ_t = exp(−∫₀ᵗ β_s ds)

    Args:
        beta_min: Initial continuous noise rate.
        beta_max: Final continuous noise rate.

    Note:
        These are continuous-time rates, not the per-step ``1e-4`` to
        ``0.02`` betas commonly used by a 1000-step discrete DDPM.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        if beta_min <= 0.0:
            raise ValueError("beta_min must be positive")
        if beta_max < beta_min:
            raise ValueError("beta_max must be greater than or equal to beta_min")
        self.beta_min = beta_min
        self.beta_max = beta_max
        log.debug("LinearDiffusionSchedule: β_min=%.3f  β_max=%.3f", beta_min, beta_max)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise rate β_t."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ_t = exp(−0.5·(β_max−β_min)·t² − β_min·t)."""
        return torch.exp(
            -0.5 * (self.beta_max - self.beta_min) * t**2 - self.beta_min * t
        )


# ---------------------------------------------------------------------------
# Diffusion-coefficient schedule g(t) for stochastic flow-matching sampling
# ---------------------------------------------------------------------------

def get_gt(
    t: torch.Tensor,
    mode: str = "1/t",
    param: float = 1.0,
    clamp_val: float | None = None,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Diffusion coefficient g(t) for stochastic (SDE) rectified-flow sampling.

    Ported from SeqFlow_v4 / Proteina. Used together with
    :meth:`LinearConditionalProbabilityPath.velocity_to_score` to turn a
    deterministic flow into a reverse SDE:
        dx = (v + g(t)·s)·dt + sqrt(2·g(t)·η·dt)·ε

    Args:
        t:         Times in [0, 1), shape (nsteps,) or broadcastable.
        mode:      "us" → (1−t)/t, "tan" → (π/2)·tan((1−t)·π/2), "1/t" → 1/t.
        param:     Power for the optional log-sigmoid reshaping (1.0 = identity).
        clamp_val: Upper clamp on g(t) (None = no upper clamp).
        eps:       Numerical stabiliser in the denominator.

    Returns:
        g(t) tensor, same shape as ``t``.
    """
    def transform_gt(gt: torch.Tensor, f_pow: float = 1.0) -> torch.Tensor:
        # Reshape the schedule via a normalised log-sigmoid power transform.
        if f_pow == 1.0:
            return gt
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        centered = log_gt - mean_log_gt
        normalized = torch.sigmoid(centered) ** f_pow
        rec = torch.logit(normalized, eps=1e-6) + mean_log_gt
        return torch.exp(rec)

    t = torch.clamp(t, 0.0, 1.0 - 1e-5)
    if mode == "us":
        gt = (1.0 - t) / (t + eps)
    elif mode == "tan":
        gt = (torch.pi / 2.0) * torch.sin((1.0 - t) * torch.pi / 2.0) / (
            torch.cos((1.0 - t) * torch.pi / 2.0) + eps
        )
    elif mode == "1/t":
        gt = 1.0 / (t + eps)
    else:
        raise NotImplementedError(f"Unknown gt mode: {mode!r}. Choose 'us', 'tan', or '1/t'.")

    gt = transform_gt(gt, f_pow=param)
    gt = torch.clamp(gt, min=0.0, max=clamp_val)  # max=None → no upper clamp
    return gt


class CosineDiffusionSchedule(DiffusionNoiseSchedule):
    """
    Cosine noise schedule (Nichol & Dhariwal, Improved DDPM, 2021).

    ᾱ_t = cos²((t/T + s) / (1 + s) · π/2) / cos²(s / (1 + s) · π/2)

    Args:
        s: Small offset to avoid singularity at t=0 (default: 0.008)
    """

    def __init__(self, s: float = 0.008):
        if not 0.0 <= s < 1.0:
            raise ValueError("s must satisfy 0 <= s < 1")
        self.s = s
        self._cos0 = torch.cos(torch.tensor(s / (1.0 + s) * torch.pi / 2)).item() ** 2
        log.debug("CosineDiffusionSchedule: s=%.4f", s)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        inner = (t + self.s) / (1.0 + self.s) * torch.pi / 2
        return (torch.cos(inner) ** 2 / self._cos0).clamp(0.0, 1.0)
