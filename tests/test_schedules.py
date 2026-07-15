import unittest

import torch

from sdelib import (
    Alpha,
    CosineDiffusionSchedule,
    LinearAlpha,
    LinearBeta,
    LinearDiffusionSchedule,
)


class QuadraticAlpha(Alpha):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t.square()


class ScheduleTest(unittest.TestCase):
    def test_flow_schedule_boundaries(self) -> None:
        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        self.assertEqual(LinearAlpha()(zero).item(), 0.0)
        self.assertEqual(LinearAlpha()(one).item(), 1.0)
        self.assertEqual(LinearBeta()(zero).item(), 1.0)
        self.assertEqual(LinearBeta()(one).item(), 0.0)

    def test_continuous_linear_schedule_reaches_noise(self) -> None:
        schedule = LinearDiffusionSchedule()
        times = torch.linspace(0.0, 1.0, 101)
        alpha_bar = schedule.alpha_bar(times)

        torch.testing.assert_close(alpha_bar[0], torch.tensor(1.0))
        self.assertTrue(torch.all(alpha_bar[1:] < alpha_bar[:-1]))
        self.assertLess(alpha_bar[-1].item(), 1e-4)

    def test_automatic_schedule_derivative_preserves_time_shape(self) -> None:
        schedule = QuadraticAlpha()
        time = torch.tensor([0.2, 0.7]).reshape(2, 1, 1)
        derivative = schedule.dt(time)

        self.assertEqual(derivative.shape, time.shape)
        torch.testing.assert_close(derivative, 2.0 * time)

    def test_cosine_schedule_boundaries(self) -> None:
        schedule = CosineDiffusionSchedule()
        values = schedule.alpha_bar(torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(values, torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0.0)

    def test_invalid_continuous_linear_rates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_min=0.0)
        with self.assertRaises(ValueError):
            LinearDiffusionSchedule(beta_min=1.0, beta_max=0.5)


if __name__ == "__main__":
    unittest.main()
