import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# optional wandb
try:
    from wandb.integration.sb3 import WandbCallback
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


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


class RacelineFeatureWrapper(gym.ObservationWrapper):
    """
    Adds 'raceline_feat' = [u, n, psi_err, v_ref] to observation dict.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "RacelineFeatureWrapper requires Dict obs"
        new_spaces = dict(env.observation_space.spaces)
        new_spaces["raceline_feat"] = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        u = 0.0; n = 0.0; psi_err = 0.0; v_ref = 0.0
        try:
            if isinstance(obs, dict):
                agent = next(iter(obs))
                frenet = obs[agent].get("frenet_pose", None)
                if frenet is not None and len(frenet) >= 3:
                    u = float(frenet[0]); n = float(frenet[1]); psi_err = float(frenet[2])
                v_ref = self._nearest_vref()
        except Exception:
            pass
        obs["raceline_feat"] = np.array([u, n, psi_err, v_ref], dtype=np.float32)
        return obs

    def _nearest_vref(self) -> float:
        try:
            envu = self.env.unwrapped
            if hasattr(envu, "track") and envu.track is not None and envu.track.raceline is not None:
                agent = envu.sim.agents[0]
                x = float(agent.state[0]); y = float(agent.state[1])
                xs = envu.track.raceline.xs; ys = envu.track.raceline.ys; vxs = envu.track.raceline.vxs
                idx = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
                return float(vxs[idx])
        except Exception:
            return 0.0
        return 0.0


class FollowingRewardWrapper(gym.Wrapper):
    """Raceline-following reward: progress along u, penalize |n|, speed tracking, collision penalty."""
    def __init__(self, env):
        super().__init__(env)
        self._last_u = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_u = self._extract_u(obs)
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        u = self._extract_u(obs); n = self._extract_n(obs); vx = self._extract_vx(obs)
        steer = self._extract_steer(action); v_ref = self._extract_vref(obs)
        shaped = 0.0
        if u is not None and self._last_u is not None:
            shaped += 5.0 * (u - self._last_u)
        if n is not None:
            shaped += -0.2 * abs(n)
        if vx is not None and steer is not None and vx < 1.0:
            shaped += -0.02 * abs(steer)
        if vx is not None and v_ref is not None:
            shaped += -0.01 * abs(vx - v_ref)
        if self._extract_collision(obs):
            shaped += -10.0
        self._last_u = u
        return obs, float(base_reward) + shaped, terminated, truncated, info

    def _extract_u(self, obs):
        try:
            if isinstance(obs, dict):
                agent = next(iter(obs)); fr = obs[agent].get("frenet_pose", None)
                if fr is not None:
                    return float(fr[0])
        except Exception:
            pass
        return None

    def _extract_n(self, obs):
        try:
            if isinstance(obs, dict):
                agent = next(iter(obs)); fr = obs[agent].get("frenet_pose", None)
                if fr is not None:
                    return float(fr[1])
        except Exception:
            pass
        return None

    def _extract_vx(self, obs):
        try:
            if isinstance(obs, dict):
                agent = next(iter(obs)); std = obs[agent].get("std_state", None)
                if std is not None and len(std) >= 4:
                    return float(std[3])
        except Exception:
            pass
        return None

    def _extract_steer(self, action):
        try:
            return float(action[1])  # after ActionOrderWrapper: [longitudinal, steer]
        except Exception:
            return None

    def _extract_collision(self, obs):
        try:
            if isinstance(obs, dict):
                agent = next(iter(obs)); col = obs[agent].get("collision", 0)
                return bool(int(col))
        except Exception:
            pass
        return False

    def _extract_vref(self, obs):
        try:
            if isinstance(obs, dict) and "raceline_feat" in obs:
                return float(obs["raceline_feat"][3])
        except Exception:
            pass
        return None


class FlattenObservation(gym.ObservationWrapper):
    """Robust flattener for nested Dict/Tuple/Box/Discrete spaces."""
    def __init__(self, env):
        super().__init__(env)
        def size_of_space(space: spaces.Space) -> int:
            if isinstance(space, spaces.Box):
                if space.shape is None or space.shape == ():
                    return 1
                return int(np.prod(space.shape))
            if isinstance(space, spaces.Discrete):
                return 1
            if isinstance(space, spaces.MultiBinary):
                shape = getattr(space, "shape", None)
                if shape is None or shape == ():
                    return int(space.n)
                return int(np.prod(shape))
            if isinstance(space, spaces.MultiDiscrete):
                return int(len(space.nvec))
            if isinstance(space, spaces.Tuple):
                return int(sum(size_of_space(s) for s in space.spaces))
            if isinstance(space, spaces.Dict):
                return int(sum(size_of_space(space.spaces[k]) for k in sorted(space.spaces.keys())))
            return 1
        self._space = env.observation_space
        total_size = size_of_space(self._space)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(total_size),), dtype=np.float32)

    def observation(self, obs):
        flat_values = []
        def append_flat(space: spaces.Space, value):
            if isinstance(space, spaces.Box):
                flat_values.extend(np.array(value, dtype=np.float32).flatten()); return
            if isinstance(space, spaces.Discrete):
                flat_values.append(float(value)); return
            if isinstance(space, spaces.MultiBinary) or isinstance(space, spaces.MultiDiscrete):
                flat_values.extend(np.array(value, dtype=np.float32).flatten()); return
            if isinstance(space, spaces.Tuple):
                for s, v in zip(space.spaces, value): append_flat(s, v); return
            if isinstance(space, spaces.Dict):
                for k in sorted(space.spaces.keys()): append_flat(space.spaces[k], value[k]); return
            flat_values.append(float(value))
        append_flat(self._space, obs)
        return np.asarray(flat_values, dtype=np.float32)


# toggle this to train or evaluate
train = True

if train:
    if _WANDB:
        run = wandb.init(project="f1tenth_gym_ppo", sync_tensorboard=True, save_code=True)
        tb_log = f"runs/{run.id}"
    else:
        run = None
        tb_log = f"runs/local_ppo"

    base_env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spa",
            "num_agents": 1,
            "timestep": 0.01,
            "lidar_num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "direct"},
            "reset_config": {"type": "rl_random_static"},
            "compute_frenet": True,
        },
    )
    # Wrap: monitor -> reward -> raceline feat -> action order -> flatten -> vec + normalize
    def _wrap():
        env = Monitor(base_env)
        env = FollowingRewardWrapper(env)
        env = RacelineFeatureWrapper(env)
        env = ActionOrderWrapper(env)
        env = FlattenObservation(env)
        return env
    env = DummyVecEnv([_wrap])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # will be faster on cpu
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=tb_log, 
        device="cpu", 
        seed=42,
        learning_rate=5e-4,
        n_steps=4096,
        batch_size=64,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
    )
    
    if _WANDB and run is not None:
        model.learn(
            total_timesteps=1_000_000,
            callback=WandbCallback(gradient_save_freq=0, model_save_path=f"models/{run.id}", verbose=2),
        )
        run.finish()
    else:
        model.learn(total_timesteps=1_000_000)
    # save VecNormalize stats
    env.save(os.path.join("models", (run.id if (_WANDB and run) else "local_ppo"), "vecnormalize.pkl"))

else:
    model_path = os.path.join(os.path.dirname(__file__), "models", "YOUR_RUN_ID", "model.zip")
    model = PPO.load(model_path, print_system_info=True, device="cpu")
    
    eval_base = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spa",
            "num_agents": 1,
            "timestep": 0.01,
            "lidar_num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "direct"},
            "reset_config": {"type": "rl_random_static"},
            "compute_frenet": True,
        },
        render_mode="human",
    )
    # same wrapper pipeline as training, then VecNormalize with loaded stats
    def _wrap_eval():
        e = Monitor(eval_base)
        e = FollowingRewardWrapper(e)
        e = RacelineFeatureWrapper(e)
        e = ActionOrderWrapper(e)
        e = FlattenObservation(e)
        return e
    eval_env = DummyVecEnv([_wrap_eval])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # try to load stats
    # infer stats dir
    stats_dir = os.path.join("models", (globals().get('run').id if (_WANDB and globals().get('run')) else "local_ppo"))
    stats_path = os.path.join(stats_dir, "vecnormalize.pkl")
    if os.path.exists(stats_path):
        try:
            eval_env.load(stats_path)
        except Exception:
            pass
    
    obs, info = eval_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        eval_env.render()

    eval_env.close()