import f1tenth_gym
import gymnasium as gym

for env_id in gym.envs.registry.keys():
    print(env_id)
