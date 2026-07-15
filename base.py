"""
Core abstract base classes for sdelib.

Defines interfaces for ODEs, SDEs, simulators, and probability distributions.
The interfaces are intentionally small and explicit.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


def _expand_batch_time(
    value: torch.Tensor,
    reference: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Represent a scalar or per-sample time as ``(B, 1, ..., 1)``."""
    if reference.ndim < 1:
        raise ValueError("reference state must include a batch dimension")

    value = value.to(device=reference.device, dtype=dtype or value.dtype)
    batch_size = reference.shape[0]
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.shape[0] == 1 and batch_size != 1:
        value = value.expand(batch_size, *value.shape[1:])
    elif value.shape[0] != batch_size:
        raise ValueError(
            f"time batch dimension {value.shape[0]} does not match state batch {batch_size}"
        )

    if value.ndim > reference.ndim:
        raise ValueError(
            f"time rank {value.ndim} cannot exceed state rank {reference.ndim}"
        )
    return value.reshape(*value.shape, *((1,) * (reference.ndim - value.ndim)))


class ODE(ABC):
    """
    Abstract base class for Ordinary Differential Equations.
    
    Represents: dX_t = u_t(X_t) dt
    """
    
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute drift coefficient u_t(x).
        
        Args:
            xt: Current state, shape (batch_size, *dims)
            t: Time, shape () or (batch_size, 1, ...)
            **kwargs: Additional arguments (e.g., conditioning)
        
        Returns:
            Drift coefficient, same shape as xt
        """
        pass


class SDE(ABC):
    """
    Abstract base class for Stochastic Differential Equations.
    
    Represents: dX_t = u_t(X_t) dt + σ_t(X_t) dW_t
    """
    
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute drift coefficient u_t(x)."""
        pass
    
    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute diffusion coefficient σ_t(x)."""
        pass


class Simulator(ABC):
    """
    Abstract base class for numerical simulation schemes.
    """
    
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform one simulation step from time t to t+h.
        
        Args:
            xt: Current state
            t: Current time
            h: Step size
            **kwargs: Additional arguments
        
        Returns:
            Next state at time t+h
        """
        pass
    
    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Simulate from initial state to final time (returns endpoint only).
        
        Args:
            x: Initial state
            ts: Time sequence, shape (num_steps,) or (batch_size, num_steps, ...)
            **kwargs: Additional arguments
        
        Returns:
            Final state at ts[-1]
        """
        # Handle different time tensor shapes
        if ts.ndim == 1:
            # Shape: (num_steps,) - same times for all samples
            for i in range(len(ts) - 1):
                t = ts[i]
                h = ts[i + 1] - ts[i]
                x = self.step(x, t, h, **kwargs)
        else:
            # Shape: (batch_size, num_steps, ...) - different times per sample
            num_steps = ts.shape[1]
            for i in range(num_steps - 1):
                t = ts[:, i]
                h = ts[:, i + 1] - ts[:, i]
                x = self.step(x, t, h, **kwargs)
        return x
    
    @torch.no_grad()
    def simulate_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Simulate and record full trajectory.
        
        Args:
            x: Initial state, shape (batch_size, *dims)
            ts: Time sequence
        
        Returns:
            Full trajectory, shape (batch_size, num_steps, *dims)
        """
        trajectory = [x]
        
        if ts.ndim == 1:
            for i in range(len(ts) - 1):
                t = ts[i]
                h = ts[i + 1] - ts[i]
                x = self.step(x, t, h, **kwargs)
                trajectory.append(x)
        else:
            num_steps = ts.shape[1]
            for i in range(num_steps - 1):
                t = ts[:, i]
                h = ts[:, i + 1] - ts[:, i]
                x = self.step(x, t, h, **kwargs)
                trajectory.append(x)
        
        return torch.stack(trajectory, dim=1)


class Sampleable(ABC):
    """
    Abstract interface for sampleable distributions.
    """
    
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample from the distribution.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tuple of (samples, labels)
            - samples: shape (num_samples, *dims)
            - labels: shape (num_samples,) or None if unlabeled
        """
        pass
