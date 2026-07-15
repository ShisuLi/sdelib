# sdelib

`sdelib` is a small PyTorch library for learning and implementing the core
mathematics of flow matching and diffusion models.  The code favors explicit
formulas and small interfaces over a large training framework.

The project originated from exercises in MIT 6.S184, *Introduction to Flow
Matching and Diffusion Models*.  It is now being developed as a from-scratch
generative-model foundation for image and protein experiments.

## Scope

The library currently provides:

- Gaussian conditional flow matching;
- straight-line rectified flow;
- continuous-time DDPM-style forward noising;
- Euler, Heun, Euler-Maruyama, and DDIM sampling;
- classifier-free guidance;
- a compact conditional U-Net for image experiments;
- small reference trainers for checking objectives.

Experiment configuration, datasets, distributed training, checkpointing,
logging, and domain-specific protein models belong in the downstream training
project rather than this library.

## Time conventions

Two directions are used, so keeping their meaning explicit is important.

### Flow matching

```text
t = 0: simple source distribution, usually Gaussian noise
t = 1: data distribution
```

For the linear path:

```text
x_t = (1 - t) x_0 + t x_1
u_t(x_t | x_1) = (x_1 - x_t) / (1 - t)
```

The conditional vector-field formula is singular at exactly `t = 1`.  Training
samples `t` from `[0, 1)`, and a solver reaches the endpoint without evaluating
the field there.

### Diffusion

```text
t = 0: clean data
t = 1: noise
```

The forward marginal is:

```text
q(x_t | x_0) = Normal(sqrt(alpha_bar(t)) x_0,
                      (1 - alpha_bar(t)) I)
```

DDIM sampling runs in decreasing time.  For schedules with
`alpha_bar(1) = 0`, start slightly below one, for example:

```python
times = torch.linspace(1.0 - 1e-4, 0.0, 51)
```

## Components

```text
base.py        ODE, SDE, Simulator, and Sampleable interfaces
schedules.py   flow interpolation and diffusion noise schedules
paths.py       Gaussian, linear, and diffusion probability paths
simulators.py  Euler, Heun, Euler-Maruyama, and DDIM methods
processes.py   classifier-free-guided vector field
models.py      time embedding and conditional U-Net
trainers.py    minimal objective implementations
```

The reference trainers deliberately remain small.  They are useful for
verifying the objective, but full experiments should use the trainer in the
downstream project.

## Environment

The server baseline is Python 3.12 with PyTorch 2.6.0 and CUDA 12.4.  Project
metadata and the CUDA wheel source are recorded in `pyproject.toml`.

When you are ready to create the environment yourself:

```bash
cd /mnt/afs/home/lishisu/projects/sdelib
uv sync --group dev
```

No dataset or model weight is downloaded by this command.

## Tests

After the editable environment is available:

```bash
uv run python -m unittest discover -s tests -v
```

The current tests focus on equations and tensor contracts:

- schedule boundaries and monotonicity;
- analytic ODE solver cases;
- per-sample time broadcasting;
- reconstruction from diffusion prediction targets;
- image and sequence-shaped trainer inputs.

## Minimal conditional flow example

```python
import torch

from sdelib import (
    CFGTrainer,
    CFGVectorFieldODE,
    EulerSimulator,
    GaussianConditionalProbabilityPath,
    IsotropicGaussian,
    LinearAlpha,
    LinearBeta,
    UNet,
)

path = GaussianConditionalProbabilityPath(
    p_data=my_image_distribution,
    p_simple=IsotropicGaussian((1, 32, 32)),
    alpha=LinearAlpha(),
    beta=LinearBeta(),
)

model = UNet(
    channels=(32, 64, 128),
    num_classes=11,  # ten classes plus the null condition
)

trainer = CFGTrainer(path, model, eta=0.1, device="cuda")
losses = trainer.train(num_epochs=1000, lr=1e-3, batch_size=64)

labels = torch.arange(10, device="cuda")
noise, _ = path.p_simple.sample(len(labels))
times = torch.linspace(0.0, 1.0, 100, device="cuda")

ode = CFGVectorFieldODE(model, guidance_scale=3.0)
samples = EulerSimulator(ode).simulate(noise, times, y=labels)
```

`num_epochs` in the minimal trainer currently means optimizer steps, not full
passes over a dataset.  The downstream training project will use a conventional
DataLoader-based trainer.

## Design rules

1. Add a mathematical abstraction only after it is needed by a concrete task.
2. Keep data loading and experiment orchestration outside `sdelib`.
3. Test endpoint behavior, broadcasting, dtype, and device movement explicitly.
4. Do not hide singularities with arbitrary clamps when a clear input contract
   can reject an invalid time.
5. Prefer one readable implementation of an equation over multiple optimized
   variants until profiling shows a real bottleneck.

## License status

The repository does not yet declare a redistribution license.  Before public
release or reuse outside the current educational work, the license terms of the
original course material must be checked and an explicit compatible license
must be added.
