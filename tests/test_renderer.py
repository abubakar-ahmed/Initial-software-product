import unittest
import numpy as np
from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.utils import deep_update


class TestRenderer(unittest.TestCase):
    @staticmethod
    def _make_env(config=None, render_mode=None) -> F110Env:
        import gymnasium as gym
        import f1tenth_gym

        base_config = {
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
        }
        config = deep_update(base_config, config or {})
        return gym.make("f1tenth_gym:f1tenth-v0", config=config, render_mode=render_mode)

    @unittest.skip("Human rendering is not supported in CI environments")
    def test_human_render(self):
        env = self._make_env(render_mode="human")
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            self.assertTrue(env.action_space.contains(action))
            env.step(action)
            env.render()
        env.close()

    def test_rgb_array_render(self):
        env = self._make_env(render_mode="rgb_array")
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)
            self.assertEqual(frame.shape[2], 3)
        env.close()

    def test_rgb_array_list(self):
        steps = 100
        env = self._make_env(render_mode="rgb_array_list")
        env.reset()
        for _ in range(steps):
            action = env.action_space.sample()
            env.step(action)

        frame_list = env.render()
        expected_frames = steps + 1

        self.assertIsInstance(frame_list, list)
        self.assertEqual(len(frame_list), expected_frames)
        for frame in frame_list:
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)
            self.assertEqual(frame.shape[2], 3)

        env.close()
