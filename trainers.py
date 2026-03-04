"""
Training utilities for flow matching and diffusion - Production Version

Trainers:
  - CFGTrainer            : Conditional flow matching with classifier-free guidance
  - DiffusionTrainer      : DDPM-style denoising (epsilon / x0 / v prediction)
  - RectifiedFlowTrainer  : Rectified / straight-line flow matching
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def model_size_mb(model: torch.nn.Module) -> float:
    """Return model parameter + buffer size in MiB."""
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / (1024**2)


class Trainer(ABC):
    """
    Base trainer class.

    Args:
        model:  Neural network to train.
        device: Target device string (inferred from model parameters if None).
    """

    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = self._resolve_device(model, device)

    @staticmethod
    def _resolve_device(model: torch.nn.Module, device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        """Compute training loss for one batch."""
        pass

    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Build Adam optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, lr: float = 1e-3, **kwargs):
        """
        Train the model.

        Args:
            num_epochs: Number of gradient steps.
            lr:         Learning rate.
            **kwargs:   Forwarded to ``get_train_loss``.

        Returns:
            List[float] of per-step losses.
        """
        log.info(
            "Starting training | steps=%d  lr=%.1e  model_size=%.2f MiB  device=%s",
            num_epochs, lr, model_size_mb(self.model), self.device,
        )
        self.model.to(self.device)
        opt = self.get_optimizer(lr)
        self.model.train()

        losses: list[float] = []
        pbar = tqdm(range(num_epochs), desc="Training")
        for idx in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_description(f"step {idx:05d}  loss={loss.item():.4f}")

        self.model.eval()
        log.info("Training complete | final_loss=%.4f", losses[-1])
        return losses


class CFGTrainer(Trainer):
    """
    Classifier-free guidance trainer for conditional flow matching.

    Implements CFG training objective with label dropping:
        L_CFM(θ) = E[ ‖u_t^θ(x|y) − u_t^{ref}(x|z)‖² ]

    Labels are randomly replaced with ``null_label`` (∅) with probability η.

    Args:
        path:       GaussianConditionalProbabilityPath instance.
        model:      Conditional vector field network u_t^θ(x, t, y).
        eta:        Label-dropout probability (0 < η < 1).
        null_label: Null label index for unconditional training (default: 10).
        device:     Target device string.
    """

    def __init__(self, path, model, eta: float, null_label: int = 10, device: Optional[str] = None):
        assert 0 < eta < 1, "eta must be in (0, 1)"
        super().__init__(model, device=device)
        self.eta = eta
        self.null_label = null_label
        self.path = path.to(self.device)
        log.info("CFGTrainer | η=%.2f  null_label=%d", eta, null_label)

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute CFG flow-matching loss for one batch.

        Args:
            batch_size: Samples per gradient step.

        Returns:
            Scalar loss tensor.
        """
        z, y = self.path.p_data.sample(batch_size)
        z = z.to(self.device)
        y = y.to(self.device)

        # Label dropout: y → ∅ with prob η
        mask = torch.rand(batch_size, device=self.device) < self.eta
        y = y.clone()
        y[mask] = self.null_label

        t = torch.rand(batch_size, 1, 1, 1, device=self.device)
        x = self.path.sample_conditional_path(z, t)

        ut_theta = self.model(x, t, y)
        ut_ref   = self.path.conditional_vector_field(x, z, t)

        # MSE averaged over spatial dims, then over batch
        return torch.sum(torch.square(ut_theta - ut_ref), dim=[1, 2, 3]).mean()


# ---------------------------------------------------------------------------
# Diffusion trainer (DDPM-style)
# ---------------------------------------------------------------------------

class DiffusionTrainer(Trainer):
    """
    DDPM-style denoising diffusion training.

    Supports three prediction targets:
        - ``"epsilon"`` : predict the added Gaussian noise ε           (Ho et al., 2020)
        - ``"x0"``      : predict the clean sample x_0
        - ``"v"``       : predict velocity v = √ᾱ·ε − √(1−ᾱ)·x_0     (Salimans, 2022)

    The denoising network has signature:
        net(x_t, t, **cond_kwargs) → prediction

    where ``t`` has shape (B, 1, 1, 1) with values in [0, 1].

    Args:
        path:   DiffusionPath instance (carries schedule + prediction_type).
        model:  Denoising network.
        device: Target device string.
    """

    def __init__(self, path, model, device: Optional[str] = None):
        super().__init__(model, device=device)
        self.path = path
        log.info(
            "DiffusionTrainer | prediction_type=%s  schedule=%s",
            path.prediction_type,
            path.schedule.__class__.__name__,
        )

    def get_train_loss(self, batch_size: int, **cond_kwargs) -> torch.Tensor:
        """
        Compute denoising loss for one batch.

        Procedure:
            1. Sample x_0 ~ p_data
            2. Sample t ~ U[0, 1] and ε ~ N(0, I)
            3. Compute x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
            4. Predict target via network; compute MSE against ground truth

        Args:
            batch_size:   Samples per gradient step.
            **cond_kwargs: Forwarded to the model (e.g. ``y=labels``).

        Returns:
            Scalar loss tensor.
        """
        x0, y = self.path.p_data.sample(batch_size)
        x0 = x0.to(self.device)

        t = torch.rand(batch_size, 1, 1, 1, device=self.device)
        xt, noise = self.path.q_sample(x0, t)

        target = self.path.get_target(x0, noise, t)

        if y is not None:
            y = y.to(self.device)
            prediction = self.model(xt, t, y=y, **cond_kwargs)
        else:
            prediction = self.model(xt, t, **cond_kwargs)

        return torch.sum(torch.square(prediction - target), dim=[1, 2, 3]).mean()


# ---------------------------------------------------------------------------
# Rectified flow trainer
# ---------------------------------------------------------------------------

class RectifiedFlowTrainer(Trainer):
    """
    Rectified flow training (Liu et al., 2022).

    Trains a velocity network v_θ on the straight-line interpolation loss:
        L(θ) = E[ ‖v_θ(x_t, t) − (z − x_0)‖² ]

    where x_t = (1−t)·x_0 + t·z,  x_0 ~ p_simple,  z ~ p_data.

    This is equivalent to ``LinearConditionalProbabilityPath``'s conditional
    vector field  u_t(x|z) = z − x_0  (constant in x for the linear path).

    Args:
        path:   LinearConditionalProbabilityPath instance.
        model:  Velocity network v_θ(x_t, t).
        device: Target device string.
    """

    def __init__(self, path, model, device: Optional[str] = None):
        super().__init__(model, device=device)
        self.path = path.to(self.device)
        log.info("RectifiedFlowTrainer | path=%s", path.__class__.__name__)

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        """
        Compute rectified flow loss for one batch.

        Args:
            batch_size: Samples per gradient step.

        Returns:
            Scalar loss tensor.
        """
        z, _ = self.path.p_data.sample(batch_size)
        z = z.to(self.device)

        x0, _ = self.path.p_simple.sample(batch_size)
        x0 = x0.to(self.device)

        t = torch.rand(batch_size, 1, 1, 1, device=self.device)
        xt = (1.0 - t) * x0 + t * z          # linear interpolation
        target = z - x0                        # constant velocity

        prediction = self.model(xt, t)

        return torch.sum(torch.square(prediction - target), dim=[1, 2, 3]).mean()
