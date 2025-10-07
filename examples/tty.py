import os
import time 
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from examples.train_ppo import make_env  # reuse training env wrappers

def make_f1_env(render_mode="human"):
    """
    Create evaluation env using the SAME wrapper stack as training.
    """
    # Use the Monza map to match the training script defaults
    return make_env(rank=0, seed=0, render_mode=render_mode, map_name="Monza")()

def evaluate_agent(model_path=None, num_episodes=5, render=True):
    """
    Evaluate a trained F1TENTH agent.
    """
    # Match training output directory from examples/train_ppo.py
    model_dir = "models/ppo_f1tenth_monza"

    if model_path is None:
        if os.path.exists(os.path.join(model_dir, "best_model.zip")):
            model_path = os.path.join(model_dir, "best_model.zip")
        elif os.path.exists(os.path.join(model_dir, "final_model.zip")):
            model_path = os.path.join(model_dir, "final_model.zip")
        else:
            raise FileNotFoundError(f"No trained model found in {model_dir}.")

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device="cpu")

    render_mode = "human" if render else None
    eval_env = make_f1_env(render_mode=render_mode)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)

            # --- THIS IS THE FIX ---
            # These two lines make it work, just like in your test script.
            eval_env.render()
            if render:
                time.sleep(0.01)
            # ----------------------
            
            episode_reward += reward
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode finished. Steps: {steps}, Reward: {episode_reward:.2f}")

    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'='*50}\n")

    eval_env.close()
    return episode_rewards, episode_lengths

if __name__ == "__main__":
    evaluate_agent(num_episodes=3, render=True)