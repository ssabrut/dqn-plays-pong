"""
Executes and visualizes a pre-trained DQN agent in a Gymnasium environment.

This script loads a saved model, initializes the specified Atari environment with
the necessary wrappers (grayscale, resize, frame stacking), and runs the agent
for a fixed number of episodes, rendering the gameplay to the screen. It serves
as a tool to visually evaluate the performance of a trained agent.
"""

import os
import sys
import time
from typing import Tuple

# Add the parent directory to the system path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ale_py
import gymnasium as gym
import torch
from torch import nn

from core import config, model, wrappers

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

    Raises:
        ValueError: If `resize_shape` or `frame_stack_k` are not positive integers.
    """
    if not isinstance(resize_shape, int) or resize_shape <= 0:
        raise ValueError(
            f"'resize_shape' must be a positive integer, but got {resize_shape}"
        )
    if not isinstance(frame_stack_k, int) or frame_stack_k <= 0:
        raise ValueError(
            f"'frame_stack_k' must be a positive integer, but got {frame_stack_k}"
        )

    env = gym.make(env_name, render_mode="human")
    env = wrappers.GrayScaleObservation(env)
    env = wrappers.ResizeObservation(env, shape=resize_shape)
    env = wrappers.FrameStack(env, k=frame_stack_k)
    return env


def load_model(
    model_path: str, obs_shape: Tuple[int, ...], num_actions: int, device: torch.device
) -> nn.Module:
    """
    Initializes the DQN model and loads its trained weights.

    Args:
        model_path (str): The file path to the saved model state dictionary.
        obs_shape (Tuple[int, ...]): The shape of the environment's observation space.
        num_actions (int): The number of actions in the environment's action space.
        device (torch.device): The device (CPU or GPU) to load the model onto.

    Returns:
        nn.Module: The loaded and configured DQN model in evaluation mode.

    Raises:
        FileNotFoundError: If the model file does not exist at the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please run the training script first to save a model."
        )

    net = model.DQN(obs_shape, num_actions).to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def main() -> None:
    """
    Main function to set up and run the agent evaluation.
    """
    env = setup_environment(config.ENV_NAME, resize_shape=84, frame_stack_k=4)
    try:
        # These shape and action space values are only valid after wrapping
        obs_shape: Tuple[int, ...] = env.observation_space.shape
        num_actions: int = env.action_space.n

        net = load_model(config.MODEL_SAVE_PATH, obs_shape, num_actions, config.DEVICE)

        print("Playing Pong with the trained agent...")
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
                total_reward += float(reward)
                time.sleep(2e-2)  # Small delay to make rendering smoother

            print(f"Episode {i + 1}, Total Reward: {total_reward}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Ensures the environment is properly closed even if an error occurs
        env.close()
        print("Evaluation finished and environment closed.")


if __name__ == "__main__":
    main()
