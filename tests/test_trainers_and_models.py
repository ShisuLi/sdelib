import unittest

import torch

from sdelib import DiffusionPath, DiffusionTrainer, LinearDiffusionSchedule, UNet
from sdelib.trainers import _mean_flat_mse


class VectorDataset:
    def sample(self, num_samples: int):
        return torch.zeros(num_samples, 7, 4), None


class RecordingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.last_time_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.last_time_shape = tuple(t.shape)
        return torch.zeros_like(x) + self.scale


class TrainerAndModelTest(unittest.TestCase):
    def test_flat_mse_is_independent_of_non_batch_shape(self) -> None:
        prediction = torch.zeros(2, 5, 3)
        target = torch.ones_like(prediction)
        self.assertEqual(_mean_flat_mse(prediction, target).item(), 1.0)

    def test_diffusion_trainer_accepts_sequence_shaped_tensors(self) -> None:
        model = RecordingDenoiser()
        path = DiffusionPath(VectorDataset(), LinearDiffusionSchedule(), "epsilon")
        loss = DiffusionTrainer(path, model, device="cpu").get_train_loss(batch_size=3)

        self.assertEqual(model.last_time_shape, (3, 1, 1))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_unet_uses_null_label_when_condition_is_absent(self) -> None:
        model = UNet(
            channels=(4, 8),
            num_residual_layers=1,
            t_embed_dim=8,
            y_embed_dim=8,
            num_classes=11,
        ).eval()
        images = torch.randn(2, 1, 8, 8)
        times = torch.rand(2, 1, 1, 1)

        with torch.no_grad():
            output = model(images, times)

        self.assertEqual(output.shape, images.shape)


if __name__ == "__main__":
    unittest.main()
