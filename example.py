"""
Example usage of SDELib - Quick test to verify the library works.

This demonstrates the core workflow:
1. Define a simple data distribution
2. Setup conditional probability path
3. Create and train a model
4. Sample from the trained model
"""

import torch
import torch.nn as nn
from sdelib import (
    GaussianConditionalProbabilityPath,
    UNet,
    CFGTrainer,
    CFGVectorFieldODE,
    EulerSimulator,
    LinearAlpha,
    LinearBeta,
    IsotropicGaussian
)


class DummyDataset(nn.Module):
    """Dummy dataset for testing - generates random images with labels."""
    
    def __init__(self, img_size=32):
        super().__init__()
        self.img_size = img_size
    
    def sample(self, num_samples):
        """Generate random images and labels."""
        images = torch.randn(num_samples, 1, self.img_size, self.img_size)
        labels = torch.randint(0, 10, (num_samples,))
        return images, labels


def main():
    """Test the library with a minimal training loop."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Setup probability path
    print("\n1. Creating conditional probability path...")
    path = GaussianConditionalProbabilityPath(
        p_data=DummyDataset(img_size=32),
        p_simple=IsotropicGaussian([1, 32, 32]),
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)
    
    # 2. Create model
    print("2. Creating U-Net model...")
    unet = UNet(
        in_channels=1,
        out_channels=1,
        channels=[16, 32],  # Smaller for testing
        num_residual_layers=1,
        t_embed_dim=32,
        y_embed_dim=32,
        num_classes=11
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # 3. Train (just a few epochs for testing)
    print("3. Training with CFG...")
    trainer = CFGTrainer(
        path=path,
        model=unet,
        eta=0.1,
        device=device
    )
    
    losses = trainer.train(
        num_epochs=10,
        lr=1e-3,
        batch_size=8
    )
    
    print(f"   Final loss: {losses[-1]:.4f}")
    
    # 4. Sample from trained model
    print("\n4. Sampling from trained model...")
    unet.eval()
    
    ode = CFGVectorFieldODE(unet, guidance_scale=2.0)
    simulator = EulerSimulator(ode)
    
    # Generate samples for each digit
    num_samples_per_class = 2
    labels = torch.arange(10).repeat_interleave(num_samples_per_class).to(device)
    x0, _ = path.p_simple.sample(len(labels))
    x0 = x0.to(device)
    
    ts = torch.linspace(0, 1, 50).to(device)
    
    with torch.no_grad():
        x1 = simulator.simulate(x0, ts, y=labels)
    
    print(f"   Generated samples shape: {x1.shape}")
    print(f"   Sample value range: [{x1.min():.3f}, {x1.max():.3f}]")
    
    print("\n✓ Library test completed successfully!")
    
    return losses, x1


if __name__ == '__main__':
    losses, samples = main()
