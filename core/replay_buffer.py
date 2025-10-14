import random
from collections import deque
from typing import Any, Deque, Tuple

import numpy as np


class ReplayBuffer:
    """
    A fixed-size replay buffer to store experience tuples.

    This class is used in reinforcement learning to store transitions (state,
    action, reward, next_state, done) and sample them in batches to train an agent.
    Storing experiences and sampling randomly helps to break the correlation
    between consecutive samples, leading to more stable training.

    Attributes:
        buffer (Deque[Tuple]): A deque used as the underlying data structure
                               for storing experiences with a fixed maximum length.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of experiences to store in the buffer.

        Raises:
            ValueError: If `capacity` is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"'capacity' must be a positive integer, but got {capacity}."
            )
        self.buffer: Deque[Tuple[np.ndarray, Any, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        """
        Returns the current number of experiences in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return len(self.buffer)

    def store(
        self, state: Any, action: Any, reward: float, next_state: Any, done: bool
    ) -> None:
        """
        Adds a new experience to the buffer.

        Args:
            state (Any): The state observed from the environment.
            action (Any): The action taken in that state.
            reward (float): The reward received after taking the action.
            next_state (Any): The next state observed after the action.
            done (bool): A flag indicating whether the episode has terminated.
        """
        # Convert states to NumPy arrays for consistency
        state_np = np.array(state, dtype=np.float32)
        next_state_np = np.array(next_state, dtype=np.float32)
        self.buffer.append((state_np, action, reward, next_state_np, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple[np.ndarray, ...]: A tuple containing five NumPy arrays:
                                    states, actions, rewards, next_states, and dones.

        Raises:
            ValueError: If `batch_size` is not a positive integer or is greater
                        than the number of experiences currently in the buffer.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"'batch_size' must be a positive integer, but got {batch_size}."
            )
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} elements when the buffer only contains "
                f"{len(self.buffer)} elements."
            )

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_),
        )
