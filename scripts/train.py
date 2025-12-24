import os
import random
import sys
import time
from collections import deque
from typing import Deque, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ale_py
import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from core import agent, config, wrappers

# Register Atari environments
gym.register_envs(ale_py)

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


def setup_environment(env_name: str, resize_shape: int, frame_stack_k: int) -> gym.Env:
    """
    Initializes and wraps the Gymnasium environment with validation.

    Args:
        env_name (str): The name of the environment to create (e.g., 'PongNoFrameskip-v4').
        resize_shape (int): The target dimension (N x N) for the observation.
        frame_stack_k (int): The number of frames to stack.

    Returns:
        gym.Env: The fully wrapped and configured environment.

    Raises:
        gym.error.Error: If the environment cannot be created.
    """
    try:
        logger.debug(f"Creating environment: {env_name}")
        env = gym.make(env_name, render_mode="rgb_array")
        
        env = wrappers.MaxAndSkipEnv(env, skip=4)
        env = wrappers.GrayScaleObservation(env)
        env = wrappers.ResizeObservation(env, shape=resize_shape)
        env = wrappers.FrameStack(env, k=frame_stack_k)
        
        logger.success(f"Environment setup complete for {env_name}.")
        return env
    except Exception as e:
        logger.exception(f"Failed to setup environment {env_name}.")
        raise e


def apply_reward_shaping(reward: float) -> float:
    """
    Clips rewards to the range [-1.0, 1.0] to stabilize training.

    Args:
        reward (float): The raw reward from the environment.

    Returns:
        float: The clipped reward.
    """
    return max(min(reward, 1.0), -1.0)


def train_agent(
    env: gym.Env,
    pong_agent: agent.DQNAgent,
    writer: SummaryWriter,
    num_episodes: int,
) -> None:
    """
    Executes the main training loop.

    Args:
        env (gym.Env): The wrapped environment.
        pong_agent (agent.DQNAgent): The initialized DQN agent.
        writer (SummaryWriter): TensorBoard writer for metrics logging.
        num_episodes (int): Maximum number of episodes to train.
    """
    episode_rewards: Deque[float] = deque(maxlen=100)
    total_frames: int = 0
    start_time: float = time.perf_counter()

    logger.info("Starting training loop...")

    for episode in range(1, num_episodes + 1):
        try:
            state, _ = env.reset()
            episode_reward: float = 0.0
            episode_shaped_reward: float = 0.0
            
            while True:
                total_frames += 1
                
                # Action Selection
                action: int = pong_agent.select_action(state)
                
                # Environment Step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done: bool = terminated or truncated

                # Reward Shaping & Storage
                shaped_reward: float = apply_reward_shaping(float(reward))
                pong_agent.buffer.store(state, action, shaped_reward, next_state, done)
                
                state = next_state
                episode_reward += float(reward)
                episode_shaped_reward += shaped_reward

                # Learning Step
                loss: Optional[float] = pong_agent.learn()
                pong_agent.decay_epsilon(total_frames)

                # Target Network Update
                if total_frames % config.TARGET_UPDATE_FREQ == 0:
                    pong_agent.update_target_network()

                # Logging (Per Step - sampled)
                if loss is not None and total_frames % 100 == 0:
                    writer.add_scalar("Stats/Loss", loss, total_frames)
                    writer.add_scalar("Stats/Epsilon", pong_agent.epsilon, total_frames)

                if done:
                    episode_rewards.append(episode_reward)
                    break
            
            # --- Episode Logging ---
            mean_reward = np.mean(episode_rewards)
            writer.add_scalar("Reward/Original_Per_Episode", episode_reward, episode)
            writer.add_scalar("Reward/Mean_Last_100", mean_reward, episode)

            if episode % config.LOG_FREQ == 0:
                fps = total_frames / (time.perf_counter() - start_time)
                logger.info(
                    f"Episode {episode:04d} | Frames {total_frames} | "
                    f"Mean Reward {mean_reward:.2f} | Epsilon {pong_agent.epsilon:.4f} | "
                    f"FPS {fps:.2f}"
                )

            # Stopping Condition (Frame Limit)
            if total_frames > config.TOTAL_FRAMES:
                logger.warning(f"Total frame limit ({config.TOTAL_FRAMES}) reached.")
                break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            break
        except Exception as e:
            logger.exception(f"Error occurred during episode {episode}.")
            raise e


def main() -> None:
    """
    Main entry point for the training script.
    
    Orchestrates configuration, initialization, training, and teardown.
    """
    # Create run directory
    run_name = f"pong_run_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    env: Optional[gym.Env] = None

    try:
        logger.info(f"Device set to: {config.DEVICE}")
        set_seed(42)

        # Environment Setup
        env = setup_environment(
            config.ENV_NAME, 
            resize_shape=84, 
            frame_stack_k=4
        )

        # Agent Initialization
        pong_agent = agent.DQNAgent(env, config)
        
        # Start Training
        train_agent(
            env=env,
            pong_agent=pong_agent,
            writer=writer,
            num_episodes=5000
        )

        # Save Model
        logger.info("Saving trained model...")
        save_dir = os.path.dirname(config.MODEL_SAVE_PATH)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        torch.save(pong_agent.policy_net.state_dict(), config.MODEL_SAVE_PATH)
        logger.success(f"Model saved successfully to {config.MODEL_SAVE_PATH}")

    except Exception:
        logger.exception("Critical failure in main execution loop.")
        sys.exit(1)
    finally:
        logger.info("Cleaning up resources...")
        if writer:
            writer.close()
        if env:
            env.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()