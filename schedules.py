"""
Time schedules for conditional probability paths - Production Version

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
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)), \
            "Alpha schedule must satisfy α_0 = 0"
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)), \
            "Alpha schedule must satisfy α_1 = 1"
    
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
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)), \
            "Beta schedule must satisfy β_0 = 1"
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)), \
            "Beta schedule must satisfy β_1 = 0"
    
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
    DDPM linear β schedule (Ho et al., 2020).

    β_t = β_min + (β_max − β_min) · t / T  (continuous approximation)
    ᾱ_t = exp(−∫₀ᵗ β_s ds)

    Args:
        beta_min: Minimum β value (default: 1e-4 per original DDPM)
        beta_max: Maximum β value (default: 0.02  per original DDPM)
    """

    def __init__(self, beta_min: float = 1e-4, beta_max: float = 0.02):
        self.beta_min = beta_min
        self.beta_max = beta_max
        log.debug("LinearDiffusionSchedule: β_min=%.1e  β_max=%.2f", beta_min, beta_max)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise rate β_t."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ_t = exp(−0.5·(β_max−β_min)·t² − β_min·t)."""
        return torch.exp(
            -0.5 * (self.beta_max - self.beta_min) * t**2 - self.beta_min * t
        )


class CosineDiffusionSchedule(DiffusionNoiseSchedule):
    """
    Cosine noise schedule (Nichol & Dhariwal, Improved DDPM, 2021).

    ᾱ_t = cos²((t/T + s) / (1 + s) · π/2) / cos²(s / (1 + s) · π/2)

    Args:
        s: Small offset to avoid singularity at t=0 (default: 0.008)
    """

    def __init__(self, s: float = 0.008):
        self.s = s
        self._cos0 = torch.cos(torch.tensor(s / (1.0 + s) * torch.pi / 2)).item() ** 2
        log.debug("CosineDiffusionSchedule: s=%.4f", s)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        inner = (t + self.s) / (1.0 + self.s) * torch.pi / 2
        return (torch.cos(inner) ** 2 / self._cos0).clamp(0.0, 1.0)
