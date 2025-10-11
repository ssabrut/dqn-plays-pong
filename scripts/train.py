import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ale_py
import torch
import time
import gymnasium as gym
import numpy as np

from core import wrappers, agent, config
gym.register_envs(ale_py)

if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")

    # Create and wrap the environment
    env = gym.make(config.ENV_NAME, render_mode="human")
    env = wrappers.GrayScaleObservation(env)
    env = wrappers.ResizeObservation(env, shape=84)
    env = wrappers.FrameStack(env, k=4)

    # Initialize agent
    pong_agent = agent.DQNAgent(env, config)

    episode_rewards = []
    total_frames = 0
    start_time = time.perf_counter()

    print("Starting training...")
    for episode in range(1, 10001):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            total_frames += 1
            action = pong_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            pong_agent.buffer.store(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            loss = pong_agent.learn()
            pong_agent.decay_epsilon(total_frames)

            if total_frames % config.TARGET_UPDATE_FREQ == 0:
                pong_agent.update_target_network()

            if done:
                episode_rewards.append(episode_reward)
                break

        print(f"Episode {episode} - Reward: {episode_reward}")

        if episode % config.LOG_FREQ == 0:
            mean_reward = np.mean(episode_rewards[-100:])
            fps = total_frames / (time.perf_counter() - start_time)
            print(f"Episode: {episode} - Frames: {total_frames} - Mean Reward: {mean_reward:.2f} - Epsilon: {pong_agent.epsilon:.4f} - FPS: {fps:.2f}")

        if total_frames > config.TOTAL_FRAMES:
            print("Training finished")
            break

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(pong_agent.policy_net.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    env.close()