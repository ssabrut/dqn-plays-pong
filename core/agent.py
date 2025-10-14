import random
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from core.config import DEVICE
from core.model import DQN
from core.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Implements a Deep Q-Network agent for reinforcement learning.

    This agent interacts with an environment, stores experiences in a replay buffer,
    and learns a policy to maximize cumulative rewards using a policy network and a
    target network for stabilized learning.

    Attributes:
        env (Any): The environment instance the agent interacts with.
        config (Any): A configuration object containing hyperparameters.
        action_space_size (int): The number of possible actions in the environment.
        obs_space_shape (Tuple[int, ...]): The shape of the observation space.
        buffer (ReplayBuffer): The replay buffer for storing experiences.
        epsilon (float): The current value for the epsilon-greedy action selection strategy.
        policy_net (DQN): The primary network for estimating Q-values.
        target_net (DQN): The target network, used for calculating target Q-values.
        optimizer (optim.Adam): The optimizer for training the policy network.
    """

    def __init__(self, env: Any, config: Any) -> None:
        """
        Initializes the DQNAgent.

        Args:
            env (Any): An environment object that must have `action_space.n` and
                       `observation_space.shape` attributes.
            config (Any): A configuration object with hyperparameters such as
                          `BUFFER_SIZE`, `EPSILON_START`, `LEARNING_RATE`, `BATCH_SIZE`,
                          `GAMMA`, `EPSILON_END`, and `EPSILON_DECAY`.

        Raises:
            AttributeError: If `env` or `config` objects are missing required attributes.
            ValueError: If a configuration value is invalid (e.g., negative buffer size).
            TypeError: If a configuration value has an incorrect type.
        """
        # --- Validation ---
        if not hasattr(env, "action_space") or not hasattr(env.action_space, "n"):
            raise AttributeError(
                "The 'env' object must have an 'action_space.n' attribute."
            )
        if not hasattr(env, "observation_space") or not hasattr(
            env.observation_space, "shape"
        ):
            raise AttributeError(
                "The 'env' object must have an 'observation_space.shape' attribute."
            )

        required_configs = [
            "BUFFER_SIZE",
            "EPSILON_START",
            "LEARNING_RATE",
            "BATCH_SIZE",
            "GAMMA",
            "EPSILON_END",
            "EPSILON_DECAY",
        ]
        for attr in required_configs:
            if not hasattr(config, attr):
                raise AttributeError(
                    f"The 'config' object is missing the required attribute: {attr}"
                )
        if not isinstance(config.BUFFER_SIZE, int) or config.BUFFER_SIZE <= 0:
            raise ValueError("'config.BUFFER_SIZE' must be a positive integer.")
        if not isinstance(config.BATCH_SIZE, int) or config.BATCH_SIZE <= 0:
            raise ValueError("'config.BATCH_SIZE' must be a positive integer.")
        if not isinstance(config.LEARNING_RATE, float) or config.LEARNING_RATE <= 0:
            raise ValueError("'config.LEARNING_RATE' must be a positive float.")
        # --- End Validation ---

        self.env: Any = env
        self.config: Any = config
        self.action_space_size: int = self.env.action_space.n
        self.obs_space_shape: Tuple[int, ...] = self.env.observation_space.shape
        self.buffer: ReplayBuffer = ReplayBuffer(self.config.BUFFER_SIZE)
        self.epsilon: float = self.config.EPSILON_START

        self.policy_net: DQN = DQN(self.obs_space_shape, self.action_space_size).to(
            DEVICE
        )
        self.target_net: DQN = DQN(self.obs_space_shape, self.action_space_size).to(
            DEVICE
        )
        self.update_target_network()
        self.target_net.eval()
        self.optimizer: optim.Adam = optim.Adam(
            self.policy_net.parameters(), lr=self.config.LEARNING_RATE, amsgrad=True
        )

    def update_target_network(self) -> None:
        """
        Copies the weights from the policy network to the target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        With probability epsilon, a random action is chosen. Otherwise, the action
        with the highest Q-value estimated by the policy network is chosen.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The selected action.

        Raises:
            TypeError: If the state is not a NumPy array.
            ValueError: If the state's shape does not match the environment's observation space.
        """
        if not isinstance(state, np.ndarray):
            raise TypeError(
                f"Input 'state' must be a NumPy array, but got {type(state)}."
            )
        if state.shape != self.obs_space_shape:
            raise ValueError(
                f"Input 'state' shape {state.shape} does not match the expected "
                f"observation space shape {self.obs_space_shape}."
            )

        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def decay_epsilon(self, frame_idx: int) -> None:
        """
        Decays the epsilon value based on the current frame index.

        The decay follows an exponential schedule.

        Args:
            frame_idx (int): The current frame number in the training process.

        Raises:
            ValueError: If frame_idx is not a non-negative integer.
        """
        if not isinstance(frame_idx, int) or frame_idx < 0:
            raise ValueError(
                f"'frame_idx' must be a non-negative integer, but got {frame_idx}."
            )

        c = self.config
        self.epsilon = c.EPSILON_END + (c.EPSILON_START - c.EPSILON_END) * np.exp(
            -1.0 * frame_idx / c.EPSILON_DECAY
        )

    def learn(self) -> Optional[float]:
        """
        Performs a single learning step.

        Samples a batch of experiences from the replay buffer, computes the loss
        using the policy and target networks, and updates the policy network's weights.
        The learning step is skipped if the buffer does not contain enough samples.

        Returns:
            Optional[float]: The loss value for the step, or None if learning was skipped.
        """
        if len(self.buffer) < self.config.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.BATCH_SIZE
        )
        states_t: torch.Tensor = torch.from_numpy(states).to(DEVICE)
        actions_t: torch.Tensor = (
            torch.from_numpy(actions).long().unsqueeze(-1).to(DEVICE)
        )
        rewards_t: torch.Tensor = torch.from_numpy(rewards).to(DEVICE)
        next_states_t: torch.Tensor = torch.from_numpy(next_states).to(DEVICE)
        dones_t: torch.Tensor = torch.from_numpy(dones).to(DEVICE)

        # Get current Q-values for the actions taken
        current_q_values: torch.Tensor = (
            self.policy_net(states_t).gather(1, actions_t).squeeze(-1)
        )

        # Compute target Q-values using the target network
        with torch.no_grad():
            # Get the maximum Q-value for the next states from the target network
            next_q_values: torch.Tensor = self.target_net(next_states_t).max(1)[0]
            # Zero out the Q-value for terminal states
            next_q_values[dones_t] = 0.0
            # Compute the target Q-value using the Bellman equation
            target_q_values: torch.Tensor = (
                rewards_t + self.config.GAMMA * next_q_values
            )

        # Compute loss and perform backpropagation
        loss: torch.Tensor = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
