#!/usr/bin/env python3
"""
Training-only script for f1tenth_gym + Stable-Baselines3 PPO.
Evaluation code removed for cleaner, faster runs.
"""

import os
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# -------------------------
# Custom wrappers & callback
# -------------------------

class RacelineFeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "RacelineFeatureWrapper requires Dict obs space"
        new_spaces = dict(env.observation_space.spaces)
        new_spaces["raceline_feat"] = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        u = n = psi_err = v_ref = 0.0
        try:
            agent = next(iter(obs))
            frenet = obs[agent].get("frenet_pose", None)
            if frenet is not None and len(frenet) >= 3:
                u, n, psi_err = frenet[:3]
            v_ref = self._nearest_vref()
        except Exception:
            pass
        obs["raceline_feat"] = np.array([u, n, psi_err, v_ref], dtype=np.float32)
        return obs

    def _nearest_vref(self):
        try:
            envu = self.env.unwrapped
            sim = getattr(envu, "sim", None)
            if sim and getattr(envu.track, "raceline", None) is not None:
                agent = sim.agents[0]
                x, y = agent.state[0], agent.state[1]
                xs, ys, vxs = envu.track.raceline.xs, envu.track.raceline.ys, envu.track.raceline.vxs
                idx = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
                return float(vxs[idx])
        except Exception:
            pass
        return 0.0


class FollowingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._last_u = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_u = self._get_u(obs)
        return obs, info

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        u = self._get_u(obs)
        n = self._get_n(obs)
        vx = self._get_vx(obs)
        v_ref = self._get_vref(obs)
        steer = action[1] if len(action) > 1 else 0.0
        collided = self._get_collision(obs)

        shaped = 0.0
        if u is not None and self._last_u is not None:
            shaped += 5.0 * (u - self._last_u)
        if n is not None:
            shaped += -0.2 * abs(n)
        if vx is not None and v_ref is not None:
            shaped += -0.01 * abs(vx - v_ref)
        if collided:
            shaped += -10.0

        self._last_u = u
        return obs, float(base_reward) + shaped, done, truncated, info

    def _get_u(self, obs):
        try:
            agent = next(iter(obs))
            return float(obs[agent]["frenet_pose"][0])
        except Exception:
            return None

    def _get_n(self, obs):
        try:
            agent = next(iter(obs))
            return float(obs[agent]["frenet_pose"][1])
        except Exception:
            return None

    def _get_vx(self, obs):
        try:
            agent = next(iter(obs))
            return float(obs[agent]["std_state"][3])
        except Exception:
            return None

    def _get_vref(self, obs):
        try:
            return float(obs.get("raceline_feat", [0, 0, 0, 0])[3])
        except Exception:
            return None

    def _get_collision(self, obs):
        try:
            agent = next(iter(obs))
            return bool(obs[agent].get("collision", 0))
        except Exception:
            return False


class ActionOrderWrapper(gym.ActionWrapper):
    """
    Expose action space as [longitudinal, steer] to the agent while env expects [steer, longitudinal].
    Clamp longitudinal to be non-negative to avoid reversing for speed control.
    """
    def __init__(self, env):
        super().__init__(env)
        orig_space = self.env.action_space
        if not isinstance(orig_space, spaces.Box) or orig_space.shape[-1] != 2:
            self._swap_supported = False
            self.action_space = orig_space
            return
        self._swap_supported = True
        low = np.array(orig_space.low, dtype=np.float32)
        high = np.array(orig_space.high, dtype=np.float32)
        new_low = low.copy(); new_high = high.copy()
        new_low[..., 0], new_low[..., 1] = low[..., 1], low[..., 0]
        new_high[..., 0], new_high[..., 1] = high[..., 1], high[..., 0]
        new_low[..., 0] = np.maximum(new_low[..., 0], 0.0)
        self.action_space = spaces.Box(low=new_low, high=new_high, shape=orig_space.shape, dtype=np.float32)

    def action(self, action):
        if not self._swap_supported:
            return action
        a = np.array(action, dtype=np.float32)
        a[..., 0] = np.maximum(a[..., 0], 0.0)
        mapped = np.empty_like(a)
        mapped[..., 0] = a[..., 1]
        mapped[..., 1] = a[..., 0]
        return mapped


class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        def size(space):
            if isinstance(space, spaces.Box):
                return int(np.prod(space.shape))
            if isinstance(space, spaces.Dict):
                return sum(size(v) for v in space.spaces.values())
            if isinstance(space, spaces.Discrete):
                return 1
            return 0
        total = size(env.observation_space)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32)
        self._space = env.observation_space

    def observation(self, obs):
        vals = []
        def flatten(space, val):
            if isinstance(space, spaces.Box):
                vals.extend(np.array(val, dtype=np.float32).flatten())
            elif isinstance(space, spaces.Dict):
                for k in sorted(space.spaces.keys()):
                    flatten(space.spaces[k], val[k])
        flatten(self._space, obs)
        return np.array(vals, dtype=np.float32)


class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            env = getattr(self, "training_env", None)
            if not isinstance(env, SubprocVecEnv):
                env.render()
                time.sleep(0.01)
        except Exception:
            pass
        return True

# -------------------------
# Environment factory
# -------------------------

def make_env(rank=0, seed=0, render_mode=None, map_name="Monza"):
    def _init():
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": map_name,
                "num_agents": 1,
                "timestep": 0.01,
                "integrator": "rk4",
                "compute_frenet": True,
            },
            render_mode=render_mode,
        )
        env = Monitor(env)
        env = FollowingRewardWrapper(env)
        env = RacelineFeatureWrapper(env)
        env = ActionOrderWrapper(env)
        env = FlattenObservation(env)
        return env
    set_random_seed(seed)
    return _init

# -------------------------
# Training function
# -------------------------

def train_agent(total_timesteps=1_000_000, n_envs=8, render_training=False, device=None):
    model_dir = "models/ppo_f1tenth_monza"
    log_dir = "runs/ppo_f1tenth_monza"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_seed = int(time.time()) & 0xFFFF
    random.seed(base_seed)
    np.random.seed(base_seed)

    if render_training:
        print("Render mode ON → forcing n_envs=1")
        n_envs = 1
        env = DummyVecEnv([make_env(rank=0, seed=base_seed, render_mode="human")])
        env_is_vecnorm = False
    else:
        print(f"Creating {n_envs} parallel environments...")
        env = SubprocVecEnv([make_env(i, seed=base_seed + i) for i in range(n_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        env_is_vecnorm = True

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // max(1, n_envs), 1),
        save_path=model_dir,
        name_prefix="ppo_model",
    )

    callbacks = [checkpoint_callback]
    if render_training:
        callbacks.append(RenderCallback())
    all_callbacks = CallbackList(callbacks)

    n_steps = 4096
    batch_size = 64
    total_rollout = n_steps * n_envs
    if total_rollout % batch_size != 0:
        raise ValueError(f"batch_size ({batch_size}) must divide n_steps * n_envs ({total_rollout})")

    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        seed=42,
        learning_rate=5e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
    )

    print("--- Starting Training ---")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=all_callbacks, progress_bar=True)
    print(f"\nTraining done in {(time.time() - start)/60:.2f} minutes")

    final_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    if env_is_vecnorm and isinstance(env, VecNormalize):
        vec_file = os.path.join(model_dir, "vecnormalize.pkl")
        env.save(vec_file)
        print(f"VecNormalize saved to {vec_file}")

    env.close()

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    train_agent(
        total_timesteps=1_000_000,
        n_envs=8,
        render_training=False,
    )
