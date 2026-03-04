# SDELib - Production-Ready Flow Matching Library

A minimal, production-ready Python library for training and sampling from flow-based generative models using classifier-free guidance (CFG). Extracted and refined from MIT 6.S184 course materials.

## Features

- **Clean API**: Minimal, well-documented interfaces
- **Production Ready**: Optimized for real model training, not just demos
- **Modular Design**: Easy to extend and customize
- **Type Hints**: Full type annotations for better IDE support
- **Efficient**: Vectorized operations, minimal overhead

## Installation

```bash
# Add to your project (assumes parent directory has torch installed)
# No external dependencies beyond PyTorch and einops
```

## Quick Start

### Training a Conditional Generative Model

```python
import torch
from sdelib import (
    GaussianConditionalProbabilityPath,
    UNet,
    CFGTrainer,
    LinearAlpha,
    LinearBeta,
    IsotropicGaussian
)

# Define your data distribution (must implement Sampleable)
class YourDataset:
    def sample(self, num_samples):
        # Return (data, labels) tuple
        # data: (num_samples, C, H, W)
        # labels: (num_samples,) or None
        pass

# Setup probability path
path = GaussianConditionalProbabilityPath(
    p_data=YourDataset(),
    p_simple=IsotropicGaussian([1, 32, 32]),  # For 32x32 grayscale
    alpha=LinearAlpha(),
    beta=LinearBeta()
)

# Create model
unet = UNet(
    in_channels=1,
    out_channels=1,
    channels=[32, 64, 128],
    num_residual_layers=2,
    t_embed_dim=40,
    y_embed_dim=40,
    num_classes=11  # 10 classes + 1 null label
)

# Train with CFG
trainer = CFGTrainer(
    path=path,
    model=unet,
    eta=0.1,  # 10% label dropout rate
    device='cuda'
)

losses = trainer.train(num_epochs=5000, lr=1e-3, batch_size=256)
```

### Sampling with Classifier-Free Guidance

```python
from sdelib import CFGVectorFieldODE, EulerSimulator

# Create CFG-guided ODE for sampling
ode = CFGVectorFieldODE(
    net=unet,
    guidance_scale=3.0  # Stronger conditioning
)

# Setup simulator
simulator = EulerSimulator(ode)

# Sample
batch_size = 64
labels = torch.randint(0, 10, (batch_size,))  # Class labels
x0, _ = path.p_simple.sample(batch_size)  # Start from noise
ts = torch.linspace(0, 1, 100)  # Time discretization

# Generate samples
x1 = simulator.simulate(x0, ts, y=labels)
```

## Architecture

### Module Structure

```
sdelib/
├── base.py          # Abstract base classes (ODE, SDE, Simulator)
├── simulators.py    # Euler and Euler-Maruyama methods
├── schedules.py     # Time schedules (α_t, β_t)
├── paths.py         # Conditional probability paths
├── models.py        # U-Net architecture
├── processes.py     # CFG-guided ODE
├── trainers.py      # Training utilities
└── __init__.py      # Public API
```

### Core Components

**Base Classes**
- `ODE`: Ordinary differential equation interface
- `SDE`: Stochastic differential equation interface
- `Simulator`: Numerical integration interface
- `Sampleable`: Probability distribution interface

**Simulators**
- `EulerSimulator`: First-order ODE solver
- `EulerMaruyamaSimulator`: First-order SDE solver

**Schedules**
- `LinearAlpha`: α_t = t
- `LinearBeta`: β_t = 1 - t
- `SquareRootBeta`: β_t = √(1 - t)

**Paths**
- `GaussianConditionalProbabilityPath`: Gaussian bridge with closed-form vector field
- `IsotropicGaussian`: N(0, I) noise source

**Models**
- `UNet`: U-Net with FiLM conditioning for image generation
- `FourierEncoder`: Time embedding layer

**Processes**
- `CFGVectorFieldODE`: Classifier-free guidance for inference

**Trainers**
- `CFGTrainer`: Efficient CFG training with label dropout

## Design Philosophy

This library focuses on:

1. **Production Use**: Real model training, not educational demos
2. **Minimalism**: Only essential, reusable components
3. **Clarity**: Clean code over clever tricks
4. **Flexibility**: Easy to extend and customize
5. **Efficiency**: Optimized for GPU training

### What's NOT Included

Removed from original lab materials:
- 2D toy datasets (circles, moons, checkerboard)
- Visualization utilities (use your own plotting)
- Educational examples and tutorials
- Langevin dynamics and score matching demos
- Non-essential SDE processes

These were great for learning but not needed for production models.

## Requirements

- PyTorch >= 2.0
- einops >= 0.8
- tqdm (for training progress bars)

## Citation

Based on course materials from:

```bibtex
@misc{flowsanddiffusions2025,
  author = {Peter Holderrieth and Ezra Erives},
  title = {Introduction to Flow Matching and Diffusion Models},
  year = {2025},
  url = {https://diffusion.csail.mit.edu/}
}
```

## License

Educational use. Please cite the original course materials.

## Example: MNIST Generation

```python
# Complete example for MNIST
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose
import torch.nn as nn

# Wrap MNIST as Sampleable
class MNISTSampler(nn.Module):
    def __init__(self, root='./data'):
        super().__init__()
        transform = Compose([Resize(32), ToTensor(), 
                            lambda x: (x - 0.5) * 2])  # Normalize to [-1, 1]
        self.dataset = MNIST(root, train=True, download=True, 
                            transform=transform)
    
    def sample(self, num_samples):
        indices = torch.randint(0, len(self.dataset), (num_samples,))
        images = torch.stack([self.dataset[i][0] for i in indices])
        labels = torch.tensor([self.dataset[i][1] for i in indices])
        return images, labels

# Train and sample as shown in Quick Start
```

## Advanced Usage

### Custom Architectures

Subclass `ConditionalVectorField` to implement your own architecture:

```python
class MyModel(ConditionalVectorField):
    def forward(self, x, t, y):
        # Your implementation
        return vector_field
```

### Custom Schedules

Implement `Alpha` or `Beta` interface:

```python
class CosineAlpha(Alpha):
    def __call__(self, t):
        return torch.sin(t * torch.pi / 2)
```

### Multi-GPU Training

```python
# Wrap model with DataParallel or DistributedDataParallel
unet = nn.DataParallel(unet)
trainer = CFGTrainer(path, unet, eta=0.1, device='cuda')
```

## Troubleshooting

**Q: Training is slow**
- Reduce batch size or model size
- Use mixed precision training (not included, add yourself)
- Check data loading bottlenecks

**Q: Generated samples are blurry**
- Increase guidance scale (try 3.0-7.0)
- Train for more epochs
- Use more sampling steps

**Q: OOM errors**
- Reduce batch_size or channels in UNet
- Use gradient checkpointing (not included)
- Reduce number of sampling timesteps

## Contributing

This is a minimal library extracted for production use. For educational materials, see the original course repository.
