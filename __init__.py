"""
Small, explicit building blocks for flow matching and diffusion models.

Supports three modern generative modelling paradigms:

  Flow Matching (Gaussian CFG)
  ────────────────────────────
  GaussianConditionalProbabilityPath + CFGTrainer + CFGVectorFieldODE + EulerSimulator

  Rectified Flow
  ──────────────
  LinearConditionalProbabilityPath + RectifiedFlowTrainer + EulerSimulator / HeunSimulator

  Diffusion (DDPM / DDIM)
  ───────────────────────
  DiffusionPath + DiffusionTrainer + DDIMSimulator
  with LinearDiffusionSchedule or CosineDiffusionSchedule

Quick Start:
    >>> from sdelib import (
    ...     GaussianConditionalProbabilityPath,
    ...     UNet, CFGTrainer, CFGVectorFieldODE,
    ...     EulerSimulator, LinearAlpha, LinearBeta,
    ... )
"""

# Base classes
from .base import ODE, SDE, Sampleable, Simulator

# Simulators
from .simulators import DDIMSimulator, EulerMaruyamaSimulator, EulerSimulator, HeunSimulator

# Schedules — flow matching
from .schedules import Alpha, Beta, LinearAlpha, LinearBeta, SquareRootBeta, get_gt

# Schedules — diffusion
from .schedules import (
    CosineDiffusionSchedule,
    DiffusionNoiseSchedule,
    LinearDiffusionSchedule,
)

# Probability paths
from .paths import (
    ConditionalProbabilityPath,
    DiffusionPath,
    GaussianConditionalProbabilityPath,
    IsotropicGaussian,
    LinearConditionalProbabilityPath,
)

# Models
from .models import ConditionalVectorField, FourierEncoder, UNet

# Processes
from .processes import CFGVectorFieldODE

# Trainers
from .trainers import CFGTrainer, DiffusionTrainer, RectifiedFlowTrainer, Trainer

__version__ = "1.1.0"

__all__ = [
    # Base
    "ODE", "SDE", "Simulator", "Sampleable",
    # Simulators
    "EulerSimulator", "EulerMaruyamaSimulator", "HeunSimulator", "DDIMSimulator",
    # Schedules — flow matching
    "Alpha", "Beta", "LinearAlpha", "LinearBeta", "SquareRootBeta", "get_gt",
    # Schedules — diffusion
    "DiffusionNoiseSchedule", "LinearDiffusionSchedule", "CosineDiffusionSchedule",
    # Paths
    "ConditionalProbabilityPath",
    "GaussianConditionalProbabilityPath",
    "LinearConditionalProbabilityPath",
    "DiffusionPath",
    "IsotropicGaussian",
    # Models
    "ConditionalVectorField", "UNet", "FourierEncoder",
    # Processes
    "CFGVectorFieldODE",
    # Trainers
    "Trainer", "CFGTrainer", "DiffusionTrainer", "RectifiedFlowTrainer",
]
