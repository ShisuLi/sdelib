import unittest

import torch

from sdelib import (
    CosineDiffusionSchedule,
    DiffusionPath,
    LinearConditionalProbabilityPath,
)


class FixedDistribution:
    def __init__(self, sample_shape: tuple[int, ...], value: float):
        self.sample_shape = sample_shape
        self.value = value

    def sample(self, num_samples: int):
        samples = torch.full((num_samples, *self.sample_shape), self.value)
        return samples, None


class PathTest(unittest.TestCase):
    def test_epsilon_prediction_recovers_clean_sample(self) -> None:
        path = DiffusionPath(None, CosineDiffusionSchedule(), "epsilon")
        clean = torch.randn(3, 5, 2)
        noise = torch.randn_like(clean)
        time = torch.full((3, 1, 1), 0.4)

        noisy, used_noise = path.q_sample(clean, time, noise=noise)
        recovered = path.predict_x0_from_net(noisy, used_noise, time)

        torch.testing.assert_close(recovered, clean, atol=1e-5, rtol=1e-5)

    def test_linear_path_and_velocity_have_expected_values(self) -> None:
        path = LinearConditionalProbabilityPath(
            p_data=FixedDistribution((4, 2), 1.0),
            p_simple=FixedDistribution((4, 2), 0.0),
        )
        target = torch.ones(3, 4, 2)
        time = torch.full((3, 1, 1), 0.25)
        state = path.sample_conditional_path(target, time)
        velocity = path.conditional_vector_field(state, target, time)

        torch.testing.assert_close(state, torch.full_like(state, 0.25))
        torch.testing.assert_close(velocity, torch.ones_like(velocity))

    def test_linear_path_rejects_terminal_singularity(self) -> None:
        path = LinearConditionalProbabilityPath(
            p_data=FixedDistribution((2,), 1.0),
            p_simple=FixedDistribution((2,), 0.0),
        )
        with self.assertRaises(ValueError):
            path.conditional_vector_field(
                torch.zeros(1, 2),
                torch.ones(1, 2),
                torch.ones(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
