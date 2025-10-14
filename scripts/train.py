"""
Main training script for the DQN agent on the Pong environment.

This script initializes the environment, sets up the DQN agent, runs the main
training loop, logs progress to TensorBoard, and saves the trained model upon
completion.
"""

import os
import sys
import time
from collections import deque
from typing import Deque

# Add the parent directory to the system path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core import agent, config, wrappers

# Register Atari environments with Gymnasium
gym.register_envs(ale_py)


def setup_environment(env_name: str, resize_shape: int, frame_stack_k: int) -> gym.Env:
    """
    Initializes and wraps the Gymnasium environment.

    Args:
        env_name (str): The name of the environment to create.
        resize_shape (int): The dimension for resizing the observation (shape x shape).
        frame_stack_k (int): The number of frames to stack.

    Returns:
        gym.Env: The fully wrapped and configured environment.
    """
    env = gym.make(env_name, render_mode="human")
    env = wrappers.GrayScaleObservation(env)
    env = wrappers.ResizeObservation(env, shape=resize_shape)
    env = wrappers.FrameStack(env, k=frame_stack_k)
    return env


def apply_reward_shaping(reward: float, action: int) -> float:
    """
    Applies a custom reward shaping heuristic for the Pong environment.

    Args:
        reward (float): The original reward from the environment.
        action (int): The action taken by the agent.

    Returns:
        float: The shaped reward.
    """
    # Positive reward for scoring, negative for being scored on
    if reward > 0:
        return 5.0
    if reward < 0:
        return -5.0

    # Small penalty for inaction (action 0 is NOOP) to encourage movement
    is_no_action = action == 0
    if is_no_action:
        return -2e-2
    # Small positive reward for any other action to encourage activity
    else:
        return 1e-2


def train_agent(
    env: gym.Env,
    pong_agent: agent.DQNAgent,
    writer: SummaryWriter,
    num_episodes: int,
) -> None:
    """
    Runs the main training loop for the DQN agent.

    Args:
        env (gym.Env): The Gymnasium environment.
        pong_agent (agent.DQNAgent): The DQN agent instance to train.
        writer (SummaryWriter): The TensorBoard writer for logging.
        num_episodes (int): The total number of episodes to run for training.
    """
    episode_rewards: Deque[float] = deque(maxlen=100)
    total_frames = 0
    start_time = time.perf_counter()

    print("Starting training...")
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_shaped_reward = 0.0

        while True:
            total_frames += 1
            action = pong_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            shaped_reward = apply_reward_shaping(reward, action)

            pong_agent.buffer.store(state, action, shaped_reward, next_state, done)
            state = next_state

            episode_reward += reward
            episode_shaped_reward += shaped_reward

            loss = pong_agent.learn()
            pong_agent.decay_epsilon(total_frames)

            if total_frames % config.TARGET_UPDATE_FREQ == 0:
                pong_agent.update_target_network()

            if done:
                episode_rewards.append(episode_reward)
                break

        # --- Logging ---
        mean_reward = np.mean(episode_rewards)
        writer.add_scalar("Reward/Original_Reward_Per_Episode", episode_reward, episode)
        writer.add_scalar("Reward/Shaped_Reward_Per_Episode", episode_shaped_reward, episode)
        writer.add_scalar("Reward/Mean_Original_Reward_Last_100", mean_reward, episode)
        writer.add_scalar("Stats/Epsilon", pong_agent.epsilon, total_frames)
        if loss is not None:
             writer.add_scalar("Stats/Loss", loss, total_frames)


        if episode % config.LOG_FREQ == 0:
            fps = total_frames / (time.perf_counter() - start_time)
            print(
                f"Episode: {episode} | Frames: {total_frames} | "
                f"Mean Reward: {mean_reward:.2f} | Epsilon: {pong_agent.epsilon:.4f} | "
                f"FPS: {fps:.2f}"
            )

        if total_frames > config.TOTAL_FRAMES:
            print(f"Reached {config.TOTAL_FRAMES} frames. Training finished.")
            break


def main() -> None:
    """
    Main function to orchestrate the training process.
    """
    print(f"Using device: {config.DEVICE}")
    writer = SummaryWriter(f"runs/pong_experiment_{int(time.time())}")
    env = setup_environment(config.ENV_NAME, resize_shape=84, frame_stack_k=4)

    try:
        pong_agent = agent.DQNAgent(env, config)
        train_agent(env, pong_agent, writer, num_episodes=1000)

        print("Saving model...")
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
        torch.save(pong_agent.policy_net.state_dict(), config.MODEL_SAVE_PATH)
        print(f"Model saved to {config.MODEL_SAVE_PATH}")

    except (Exception, KeyboardInterrupt) as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print("Closing resources...")
        writer.close()
        env.close()


if __name__ == "__main__":
    main()