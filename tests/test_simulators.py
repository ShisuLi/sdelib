import unittest

import torch

from sdelib import DDIMSimulator, DiffusionPath, EulerSimulator, HeunSimulator, ODE
from sdelib.schedules import LinearDiffusionSchedule


class ConstantODE(ODE):
    def drift_coefficient(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return torch.ones_like(xt)


class LinearODE(ODE):
    def drift_coefficient(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return xt


class ZeroDenoiser(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class SimulatorTest(unittest.TestCase):
    def test_euler_constant_ode(self) -> None:
        initial = torch.zeros(2, 3, 4)
        result = EulerSimulator(ConstantODE()).simulate(
            initial,
            torch.linspace(0.0, 1.0, 5),
        )
        torch.testing.assert_close(result, torch.ones_like(initial))

    def test_euler_supports_per_sample_time_grids(self) -> None:
        initial = torch.zeros(2, 3, 4)
        times = torch.tensor([[0.0, 0.5, 1.0], [0.0, 1.0, 2.0]])
        result = EulerSimulator(ConstantODE()).simulate(initial, times)

        torch.testing.assert_close(result[0], torch.ones_like(result[0]))
        torch.testing.assert_close(result[1], torch.full_like(result[1], 2.0))

    def test_heun_one_step_for_linear_ode(self) -> None:
        result = HeunSimulator(LinearODE()).simulate(
            torch.ones(2, 3),
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(result, torch.full_like(result, 2.5))

    def test_ddim_endpoint_matches_trajectory_and_supports_non_image_states(self) -> None:
        path = DiffusionPath(
            p_data=None,
            schedule=LinearDiffusionSchedule(),
            prediction_type="epsilon",
        )
        simulator = DDIMSimulator(ZeroDenoiser(), path)
        initial = torch.randn(2, 5, 3)
        times = torch.tensor([0.9, 0.5, 0.0])

        endpoint = simulator.simulate(initial.clone(), times)
        trajectory = simulator.simulate_trajectory(initial.clone(), times)

        self.assertEqual(endpoint.shape, initial.shape)
        self.assertEqual(trajectory.shape, (2, 3, 5, 3))
        torch.testing.assert_close(endpoint, trajectory[:, -1])

    def test_ddim_rejects_singular_start_time(self) -> None:
        path = DiffusionPath(None, LinearDiffusionSchedule(), "epsilon")
        simulator = DDIMSimulator(ZeroDenoiser(), path)
        with self.assertRaises(ValueError):
            simulator.simulate(torch.randn(2, 3), torch.tensor([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
