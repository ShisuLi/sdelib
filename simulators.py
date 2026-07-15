"""
Numerical simulators for ODEs and SDEs.

Implementations:
  - EulerSimulator          : 1st-order ODE solver (flow matching / rectified flow)
  - EulerMaruyamaSimulator  : 1st-order SDE solver (diffusion reverse SDE)
  - HeunSimulator           : 2nd-order ODE solver (two evaluations per step)
  - DDIMSimulator           : Deterministic DDIM sampler for diffusion models
"""

from __future__ import annotations

import logging

import torch

from .base import ODE, SDE, Simulator, _expand_batch_time

log = logging.getLogger(__name__)


class EulerSimulator(Simulator):
    """
    Euler method for ODEs: X_{t+h} = X_t + h · u_t(X_t)
    
    Args:
        ode: ODE system to simulate
    """
    
    def __init__(self, ode: ODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform one Euler step."""
        t = _expand_batch_time(t, xt)
        h = _expand_batch_time(h, xt, dtype=xt.dtype)
        drift = self.ode.drift_coefficient(xt, t, **kwargs)
        return xt + h * drift


class EulerMaruyamaSimulator(Simulator):
    """
    Euler-Maruyama method for SDEs:
    X_{t+h} = X_t + h · u_t(X_t) + √h · σ_t(X_t) · Z_t

    Args:
        sde: SDE system to simulate
    """

    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform one Euler-Maruyama step."""
        t = _expand_batch_time(t, xt)
        h = _expand_batch_time(h, xt, dtype=xt.dtype)
        drift = self.sde.drift_coefficient(xt, t, **kwargs)
        diffusion = self.sde.diffusion_coefficient(xt, t, **kwargs)
        # Brownian increment: √h · Z_t,  Z_t ~ N(0, I)
        noise = torch.randn_like(xt)
        sqrt_h = torch.sqrt(h.abs())
        return xt + h * drift + sqrt_h * diffusion * noise


class HeunSimulator(Simulator):
    """
    Heun's method (2nd-order Runge-Kutta) for ODEs.

    Corrects the Euler predictor with one extra function evaluation:
        X̃_{t+h} = X_t + h · u_t(X_t)              (Euler predictor)
        X_{t+h}  = X_t + h/2 · (u_t(X_t) + u_{t+h}(X̃_{t+h}))  (corrector)

    Same number of *steps* as Euler but 2× NFE.  Use this when you can
    afford slightly more compute per step for better sample quality.

    Args:
        ode: ODE system to simulate
    """

    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform one Heun step (predictor + corrector)."""
        t = _expand_batch_time(t, xt)
        h = _expand_batch_time(h, xt, dtype=xt.dtype)
        u1 = self.ode.drift_coefficient(xt, t, **kwargs)
        x_pred = xt + h * u1
        t_next = t + h
        u2 = self.ode.drift_coefficient(x_pred, t_next, **kwargs)
        return xt + 0.5 * h * (u1 + u2)


class DDIMSimulator:
    """
    Deterministic DDIM sampler for diffusion models (Song et al., 2020).

    Implements the DDIM update rule (η=0, fully deterministic):
        x̂_0     = predict_x0_from_net(x_t, net(x_t, t), t)
        ε̂       = (x_t − √ᾱ_t · x̂_0) / √(1−ᾱ_t)
        x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1−ᾱ_{t-1}) · ε̂

    The ``path`` argument must be a ``DiffusionPath`` instance (see paths.py).
    The denoising network ``net`` has signature ``net(x, t, **kwargs) → prediction``,
    where ``t`` has shape ``(B, 1, ..., 1)`` and values in [0, 1).

    Args:
        net:   Denoising network.
        path:  DiffusionPath carrying the noise schedule.
        eta:   Stochasticity (0 = DDIM deterministic, 1 = DDPM stochastic).
    """

    def __init__(self, net, path, eta: float = 0.0):
        if eta < 0.0:
            raise ValueError("eta must be non-negative")
        self.net = net
        self.path = path
        self.eta = eta

    @staticmethod
    def _validate_times(ts: torch.Tensor) -> None:
        if ts.ndim != 1 or ts.numel() < 2:
            raise ValueError("ts must be one-dimensional with at least two times")
        if not torch.all(ts[:-1] > ts[1:]):
            raise ValueError("ts must be strictly decreasing")
        if ts[0] >= 1.0 or ts[-1] < 0.0:
            raise ValueError("DDIM times must satisfy 0 <= ts[-1] < ts[0] < 1")

    def _step(
        self,
        x: torch.Tensor,
        t_now: torch.Tensor,
        t_next: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply one DDIM update for endpoint or trajectory sampling."""
        t_batch = _expand_batch_time(t_now, x)
        t_next_batch = _expand_batch_time(t_next, x)
        net_out = self.net(x, t_batch, **kwargs)

        x0_pred = self.path.predict_x0_from_net(x, net_out, t_batch)
        eps_pred = self.path.predict_noise_from_net(x, net_out, t_batch)
        schedule = self.path.schedule
        sqrt_ab_next = schedule.sqrt_alpha_bar(t_next_batch)

        if self.eta == 0.0:
            sqrt_1mab_next = schedule.sqrt_one_minus_alpha_bar(t_next_batch)
            return sqrt_ab_next * x0_pred + sqrt_1mab_next * eps_pred

        ab_now = schedule.alpha_bar(t_batch)
        ab_next = schedule.alpha_bar(t_next_batch)
        sigma = self.eta * torch.sqrt(
            ((1.0 - ab_next) / (1.0 - ab_now).clamp(min=1e-8))
            * (1.0 - ab_now / ab_next.clamp(min=1e-8)).clamp(min=0.0)
        )
        direction = torch.sqrt((1.0 - ab_next - sigma.square()).clamp(min=0.0))
        return (
            sqrt_ab_next * x0_pred
            + direction * eps_pred
            + sigma * torch.randn_like(x)
        )

    @torch.no_grad()
    def simulate(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run DDIM sampling from x_T (noisy) → x_0 (clean).

        Args:
            x:   Initial noise. Shape: (B, *sample_shape)
            ts:  Decreasing time sequence in [0, 1], shape (num_steps,).
                 For example ``torch.linspace(1 - 1e-4, 0, 51)``.
            **kwargs: Passed to the denoising network (e.g. y=labels).

        Returns:
            Samples at ``ts[-1]``, with the same shape as ``x``.
        """
        self._validate_times(ts)

        for i in range(len(ts) - 1):
            x = self._step(x, ts[i], ts[i + 1], **kwargs)

        return x

    @torch.no_grad()
    def simulate_trajectory(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run DDIM and record all intermediate states.

        Returns:
            Trajectory tensor, shape (B, num_steps, *sample_shape).
        """
        self._validate_times(ts)
        trajectory = [x]

        for i in range(len(ts) - 1):
            x = self._step(x, ts[i], ts[i + 1], **kwargs)
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)
