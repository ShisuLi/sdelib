"""
Conditional probability paths for flow matching and diffusion.

Defines probability paths p_t(x|z) for conditional generation:
  - GaussianConditionalProbabilityPath  (Gaussian flow matching)
  - LinearConditionalProbabilityPath    (rectified / straight-line flow)
  - DiffusionPath                       (DDPM / score-based diffusion)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
log = logging.getLogger(__name__)


def _require_open_unit_time(t: torch.Tensor, operation: str) -> None:
    """Reject times where formulas containing ``1 / (1 - t)`` are singular."""
    if torch.any((t < 0.0) | (t >= 1.0)):
        raise ValueError(f"{operation} requires every time to satisfy 0 <= t < 1")


class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract conditional probability path p_t(x|z).
    
    Defines interpolation from p_0(x) = p_simple to p_1(x|z) = δ_z.
    
    Args:
        p_simple: Source distribution (e.g., isotropic Gaussian)
        p_data: Target data distribution
    """
    
    def __init__(self, p_simple, p_data):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
    
    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from marginal p_t(x) = ∫ p_t(x|z) p(z) dz.
        
        Args:
            t: Time tensor, shape (batch_size, 1, 1, 1)
        
        Returns:
            Samples from marginal, shape (batch_size, *dims)
        """
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples)
        x = self.sample_conditional_path(z, t)
        return x
    
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample (z, y) from p_data.
        
        Args:
            num_samples: Number of samples
        
        Returns:
            Tuple of (conditioning_variable, labels)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x ~ p_t(x|z).
        
        Args:
            z: Conditioning variable, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
        
        Returns:
            Sample from conditional path, shape (batch_size, *dims)
        """
        pass
    
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional vector field u_t(x|z).
        
        Args:
            x: Position, shape (batch_size, *dims)
            z: Conditioning variable, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
        
        Returns:
            Vector field, shape (batch_size, *dims)
        """
        pass
    
    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional score ∇_x log p_t(x|z).
        
        Args:
            x: Position, shape (batch_size, *dims)
            z: Conditioning variable, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
        
        Returns:
            Score, shape (batch_size, *dims)
        """
        pass


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Gaussian conditional path: p_t(x|z) = N(x; α_t·z, β_t²·I)
    
    Provides closed-form vector field and score functions.
    
    Args:
        p_data: Target data distribution (must implement Sampleable)
        alpha: Alpha schedule (α_0=0, α_1=1)
        beta: Beta schedule (β_0=1, β_1=0)
        p_simple: Source distribution (overrides p_simple_shape if provided)
        p_simple_shape: Shape for isotropic Gaussian source (e.g., [1, 32, 32])
    """
    
    def __init__(self, p_data, alpha, beta, p_simple=None, p_simple_shape=None):
        # Create isotropic Gaussian if needed
        if p_simple is None:
            if p_simple_shape is None:
                raise ValueError("Must provide either p_simple or p_simple_shape")
            p_simple = IsotropicGaussian(p_simple_shape)
        
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample (z, y) from p_data."""
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x ~ N(α_t·z, β_t²·I).
        
        Args:
            z: Data sample, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
        
        Returns:
            Sample from conditional path
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        
        # Sample noise
        noise, _ = self.p_simple.sample(z.shape[0])
        noise = noise.to(device=z.device, dtype=z.dtype)
        
        # Apply Gaussian conditional: x = α_t·z + β_t·ε
        return alpha_t * z + beta_t * noise
    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Closed-form vector field: u_t(x|z) = α̇_t·z + (β̇_t / β_t)·(x − α_t·z)
        
        Args:
            x: Position, shape (batch_size, *dims)
            z: Conditioning variable, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
        
        Returns:
            Vector field u_t(x|z)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)
        if torch.any(beta_t == 0.0):
            raise ValueError("conditional vector field is undefined where beta(t) is zero")
        
        return dt_alpha_t * z + (dt_beta_t / beta_t) * (x - alpha_t * z)
    
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Closed-form score: ∇_x log p_t(x|z) = (α_t·z − x) / β_t²
        
        Args:
            x: Position, shape (batch_size, *dims)
            z: Conditioning variable, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
        
        Returns:
            Score ∇_x log p_t(x|z)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        if torch.any(beta_t == 0.0):
            raise ValueError("conditional score is undefined where beta(t) is zero")
        
        return (alpha_t * z - x) / (beta_t ** 2)

class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    Deterministic linear interpolation path (flow matching straight path).
    
    Defines the conditional:
        X_t = (1 − t)·X_0 + t·z
    
    where X_0 ~ p_simple is independent noise.  The path is deterministic
    given (X_0, z), so it does NOT have a tractable closed-form conditional
    score (see conditional_score below).
    
    Closed-form conditional vector field:
        u_t(x|z) = (z − x) / (1 − t)

    Closed-form conditional / marginal scores (source X_0 ~ N(0, σ²I)):
        conditional   p_t(x|z) = N(t·z, ((1−t)·σ)²·I)
            ∇_x log p_t(x|z) = (t·z − x) / ((1−t)²·σ²)
        marginal (from a learned velocity field v, stochastic-interpolant identity)
            v(x,t) = (1/t)·(x + σ²·(1−t)·s)  ⇒  s = (t·v − x) / (σ²·(1−t))

    Note:
        Both the vector field and the scores diverge as t → 1 and must not be
        evaluated exactly at t = 1.
    """

    def __init__(self, p_data, p_simple=None, p_simple_shape=None):
        """
        Initialize linear conditional path.

        Args:
            p_data        (Sampleable): Target data distribution p(z).
            p_simple      (Sampleable, optional): Source (noise) distribution p_0(x).
            p_simple_shape (list[int], optional): Shape for IsotropicGaussian source,
                used when p_simple is None.
        """
        if p_simple is not None:
            base = p_simple
        elif p_simple_shape is not None:
            base = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        else:
            raise ValueError(
                "LinearConditionalProbabilityPath requires either "
                "p_simple or p_simple_shape."
            )
        super().__init__(base, p_data)
        # Std of the Gaussian source X_0 ~ N(0, σ²I); used by the score formulas.
        self.sigma = float(getattr(base, "std", 1.0))
    
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample (z, y) ~ p_data.
        
        Args:
            num_samples (int): Number of samples.
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (z, y) where y may be None.
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x ~ p_t(x|z) via linear interpolation.
        
        Reparameterisation:
            x0, _ = p_simple.sample(bs)
            x_t   = (1 − t)·x0 + t·z
        
        Args:
            z (torch.Tensor): Conditioning variable.
                Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1].
                Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Linearly interpolated sample.
                Shape: (bs, c, h, w)
        """
        x0, _ = self.p_simple.sample(z.shape[0])
        x0 = x0.to(device=z.device, dtype=z.dtype)
        t = t.to(device=z.device, dtype=z.dtype)
        xt = (1 - t) * x0 + t * z
        return xt
    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the conditional vector field u_t(x|z) = (z − x) / (1 − t).
        
        Args:
            x (torch.Tensor): Current position on the path. Shape: (bs, c, h, w)
            z (torch.Tensor): Conditioning variable.       Shape: (bs, c, h, w)
            t (torch.Tensor): Time in [0, 1).              Shape: (bs, 1, 1, 1)
        
        Returns:
            torch.Tensor: Velocity pointing from x toward z.
                Shape: (bs, c, h, w)
        
        Warning:
            Undefined at t = 1; avoid evaluating exactly there.
        """
        _require_open_unit_time(t, "conditional vector field")
        t = t.to(device=x.device, dtype=x.dtype)
        return (z - x) / (1 - t)
    
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Closed-form conditional score ∇_x log p_t(x|z).

        With a Gaussian source X_0 ~ N(0, σ²I) and the linear interpolation
        x_t = (1−t)·X_0 + t·z, the conditional is Gaussian:
            p_t(x|z) = N(t·z, ((1−t)·σ)²·I)
        so its score is closed-form:
            ∇_x log p_t(x|z) = (t·z − x) / ((1−t)²·σ²)

        Args:
            x (torch.Tensor): Position.            Shape: (bs, *dims)
            z (torch.Tensor): Conditioning sample. Shape: (bs, *dims)
            t (torch.Tensor): Time in [0, 1).      Shape broadcastable to x.

        Returns:
            torch.Tensor: Conditional score, same shape as x.

        Warning:
            Diverges at t = 1; do not evaluate exactly there.
        """
        _require_open_unit_time(t, "conditional score")
        t = t.to(device=x.device, dtype=x.dtype)
        return (t * z - x) / ((1.0 - t) ** 2 * self.sigma ** 2)

    def velocity_to_score(self, x: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convert a (learned) marginal velocity field to the marginal score.

        Stochastic-interpolant identity for x_t = (1−t)·X_0 + t·X_1 with
        X_0 ~ N(0, σ²I):
            v(x,t) = (1/t)·(x + σ²·(1−t)·s)   ⇒   s = (t·v − x) / (σ²·(1−t))

        This is the relation used to drive stochastic (SDE) sampling from a
        velocity-prediction flow model (cf. SeqFlow_v4 ``vf_to_score``).

        Args:
            x (torch.Tensor): Current state x_t.       Shape: (bs, *dims)
            v (torch.Tensor): Marginal velocity v(x_t,t). Shape: (bs, *dims)
            t (torch.Tensor): Time in [0, 1).          Shape broadcastable to x.

        Returns:
            torch.Tensor: Marginal score ∇_x log p_t(x), same shape as x.

        Warning:
            Requires t < 1 (diverges at t = 1).
        """
        _require_open_unit_time(t, "velocity-to-score conversion")
        t = t.to(device=x.device, dtype=x.dtype)
        return (t * v - x) / (self.sigma ** 2 * (1.0 - t))


class IsotropicGaussian(torch.nn.Module):
    """
    Isotropic Gaussian distribution N(0, σ²·I).

    Args:
        shape: Shape of samples (excluding batch dimension), e.g., [1, 32, 32]
        std: Standard deviation (default: 1.0)
    """

    def __init__(self, shape: Sequence[int], std: float = 1.0):
        super().__init__()
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("shape must contain positive dimensions")
        if std <= 0.0:
            raise ValueError("std must be positive")
        self.shape = tuple(shape)
        self.std = std
        # Register buffer to track device
        self.register_buffer("_dummy", torch.zeros(1))

    @property
    def device(self) -> torch.device:
        """Get the device this module is on."""
        return self._dummy.device

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample from N(0, σ²·I).

        Args:
            num_samples: Number of samples

        Returns:
            Tuple of ``(samples, None)``.
            samples shape: (num_samples, *shape)
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        samples = torch.randn(num_samples, *self.shape, device=self.device) * self.std
        return samples, None


# ---------------------------------------------------------------------------
# Diffusion path — DDPM / score-based
# ---------------------------------------------------------------------------

class DiffusionPath(torch.nn.Module):
    """
    DDPM-style conditional probability path.

    Defines the forward noising process:
        q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1 − ᾱ_t) · I)

    Supports three prediction targets used by the denoising network:
        - ``"epsilon"``  : predict the added noise ε              (Ho et al., 2020)
        - ``"x0"``       : predict the clean image x_0            (alternative)
        - ``"v"``        : predict the velocity v = √ᾱ·ε − √(1−ᾱ)·x_0  (Salimans 2022)

    Args:
        p_data:    Target data distribution with ``.sample(n)`` → (x0, y).
        schedule:  DiffusionNoiseSchedule (e.g. CosineDiffusionSchedule).
        prediction_type: One of ``"epsilon"``, ``"x0"``, ``"v"``.
    """

    def __init__(self, p_data, schedule, prediction_type: str = "epsilon"):
        super().__init__()
        if prediction_type not in {"epsilon", "x0", "v"}:
            raise ValueError(
                "prediction_type must be 'epsilon', 'x0', or 'v', "
                f"got {prediction_type!r}"
            )
        self.p_data = p_data
        self.schedule = schedule
        self.prediction_type = prediction_type
        log.info(
            "DiffusionPath | prediction_type=%s | schedule=%s",
            prediction_type,
            schedule.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Forward noising
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x_0) = N(√ᾱ_t · x_0, (1−ᾱ_t)·I).

        Args:
            x0:    Clean samples. Shape: (B, *sample_shape)
            t:     Time in [0, 1], broadcastable to x0.
            noise: Optional pre-sampled epsilon with the same shape as x0.

        Returns:
            (xt, noise) — noised samples and the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.schedule.sqrt_alpha_bar(t)          # (B, 1, 1, 1)
        sqrt_1mab = self.schedule.sqrt_one_minus_alpha_bar(t)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    # ------------------------------------------------------------------
    # Training target
    # ------------------------------------------------------------------

    def get_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the prediction target for the denoising network.

        Args:
            x0:    Clean samples. Shape: (B, *sample_shape)
            noise: Sampled epsilon with the same shape as x0.
            t:     Time broadcastable to x0.

        Returns:
            target tensor, same shape as x0.
        """
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "x0":
            return x0
        # v-prediction: v = √ᾱ·ε − √(1−ᾱ)·x_0
        sqrt_ab = self.schedule.sqrt_alpha_bar(t)
        sqrt_1mab = self.schedule.sqrt_one_minus_alpha_bar(t)
        return sqrt_ab * noise - sqrt_1mab * x0

    # ------------------------------------------------------------------
    # Posterior (DDPM reverse step)
    # ------------------------------------------------------------------

    def predict_x0_from_net(
        self,
        xt: torch.Tensor,
        net_out: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover x̂_0 from the network output, regardless of prediction_type.

        Args:
            xt:      Noisy state.         Shape: (B, C, H, W)
            net_out: Network prediction.  Shape: (B, C, H, W)
            t:       Time.                Shape: (B, 1, 1, 1)

        Returns:
            Estimated x̂_0.  Shape: (B, C, H, W)
        """
        sqrt_ab = self.schedule.sqrt_alpha_bar(t)
        sqrt_1mab = self.schedule.sqrt_one_minus_alpha_bar(t)
        if self.prediction_type == "x0":
            return net_out
        if self.prediction_type == "epsilon":
            return (xt - sqrt_1mab * net_out) / sqrt_ab.clamp(min=1e-6)
        # v-prediction: x0 = √ᾱ·x_t − √(1−ᾱ)·v
        return sqrt_ab * xt - sqrt_1mab * net_out

    def predict_noise_from_net(
        self,
        xt: torch.Tensor,
        net_out: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover predicted noise ε̂ from the network output.

        Returns:
            Estimated ε̂.  Shape: (B, C, H, W)
        """
        sqrt_ab = self.schedule.sqrt_alpha_bar(t)
        sqrt_1mab = self.schedule.sqrt_one_minus_alpha_bar(t)
        if self.prediction_type == "epsilon":
            return net_out
        if self.prediction_type == "x0":
            return (xt - sqrt_ab * net_out) / sqrt_1mab.clamp(min=1e-6)
        # v-prediction: ε = √ᾱ·v + √(1−ᾱ)·x_t
        return sqrt_ab * net_out + sqrt_1mab * xt
