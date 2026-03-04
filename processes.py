"""
Process implementations for sampling - Production Version

Provides:
  - CFGVectorFieldODE : classifier-free guidance ODE for flow matching inference
"""

from __future__ import annotations

import logging

import torch

from .base import ODE

log = logging.getLogger(__name__)


class CFGVectorFieldODE(ODE):
    """
    Classifier-free guidance ODE for inference.
    
    Implements guided vector field:
        ũ_t(x|y) = (1 - w)·u_t(x|∅) + w·u_t(x|y)
    
    Args:
        net: Trained conditional vector field network u_t^θ(x|y)
        guidance_scale: Guidance strength w (typical range: 1.0-7.0)
            - w = 1.0: No guidance (standard conditional)
            - w > 1.0: Stronger conditioning
            - w = 0.0: Fully unconditional
        null_label: Null label value for unconditional (default: 10)
    """
    
    def __init__(self, net, guidance_scale: float = 1.0, null_label: int = 10):
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label
    
    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute CFG-guided drift.
        
        Args:
            x: Noisy state, shape (batch_size, *dims)
            t: Time, shape (batch_size, 1, 1, 1)
            y: Class labels, shape (batch_size,)
        
        Returns:
            Guided vector field
        """
        # Conditional: u_t(x|y)
        guided = self.net(x, t, y)
        
        # Unconditional: u_t(x|∅)
        unguided_y = torch.full_like(y, self.null_label)
        unguided = self.net(x, t, unguided_y)
        
        # CFG formula
        return (1 - self.guidance_scale) * unguided + self.guidance_scale * guided
