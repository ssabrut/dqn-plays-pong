import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ale_py
import time
import torch
import gymnasium as gym

from core import wrappers, model, config

gym.register_envs(ale_py)

if __name__ == "__main__":
    env = gym.make(config.ENV_NAME, render_mode="human")
    env = wrappers.GrayScaleObservation(env)
    env = wrappers.ResizeObservation(env, shape=84)
    env = wrappers.FrameStack(env, k=4)

    net = model.DQN(env.observation_space.shape, env.action_space.n).to(config.DEVICE)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        print("Please run train.py first to train and save a model.")
        sys.exit(1)

    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    net.load_state_dict(state_dict)
    net.eval()

    print("Playing pong with trained agent...")
    for i in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                q_values = net(state_t)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            time.sleep(2e-2)
        print(f"Episode {i+1}, Total Reward: {total_reward}")
        
    env.close()