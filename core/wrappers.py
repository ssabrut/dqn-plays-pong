from collections import deque
from typing import Any, Deque, SupportsFloat, Tuple, Union

import cv2
import gymnasium as gym
import numpy as np


class GrayScaleObservation(gym.ObservationWrapper):
    """
    A Gymnasium wrapper that converts RGB image observations to grayscale.

    This wrapper modifies the observation space to reflect the change from a
    3-channel RGB image to a single-channel grayscale image.

    Attributes:
        observation_space (gym.spaces.Box): The modified observation space with a
                                            single channel.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Initializes the GrayScaleObservation wrapper.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.

        Raises:
            ValueError: If the environment's observation space is not a Box, is not
                        defined, or does not have at least 3 dimensions (H, W, C).
        """
        super().__init__(env)

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("GrayScaleObservation is only compatible with Box observation spaces.")
        if self.observation_space.shape is None or len(self.observation_space.shape) < 3:
            raise ValueError("Observation space shape must have at least 3 dimensions (H, W, C).")

        obs_shape: Tuple[int, ...] = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Converts an RGB observation to grayscale.

        Args:
            observation (np.ndarray): The input RGB image observation.

        Returns:
            np.ndarray: The grayscaled image observation.

        Raises:
            TypeError: If the observation is not a NumPy array.
            ValueError: If the observation is not a 3D array with 3 color channels.
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError(f"Observation must be a NumPy array, but got {type(observation)}.")
        if observation.ndim != 3 or observation.shape[2] != 3:
            raise ValueError(
                "Observation must be a 3D array with 3 color channels (H, W, C) for grayscale conversion."
            )
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """
    A Gymnasium wrapper that resizes image observations to a specified shape.

    Attributes:
        shape (Tuple[int, int]): The target shape (height, width) for the observations.
    """

    def __init__(self, env: gym.Env, shape: Union[int, Tuple[int, int]]) -> None:
        """
        Initializes the ResizeObservation wrapper.

        Args:
            shape (Union[int, Tuple[int, int]]): The target shape. If an integer is
                                                 provided, it's used for both height and width.

        Raises:
            ValueError: If `shape` is not a positive integer or a tuple of two
                        positive integers. Or if the environment's observation
                        space is not defined.
        """
        super().__init__(env)

        if isinstance(shape, int):
            if shape <= 0:
                raise ValueError(f"If shape is an int, it must be positive, but got {shape}.")
            self.shape: Tuple[int, int] = (shape, shape)
        elif isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(i, int) and i > 0 for i in shape):
            self.shape = shape
        else:
            raise ValueError(f"'shape' must be a positive int or a tuple of two positive ints, but got {shape}.")

        if not self.observation_space or not hasattr(self.observation_space, "shape"):
             raise ValueError("Observation space or its shape is not defined")

        obs_shape: Tuple[int, ...] = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Resizes an observation to the target shape.

        Args:
            observation (np.ndarray): The input image observation.

        Returns:
            np.ndarray: The resized image observation.

        Raises:
            TypeError: If the observation is not a NumPy array.
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError(f"Observation must be a NumPy array, but got {type(observation)}.")
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)


class FrameStack(gym.ObservationWrapper):
    """
    A Gymnasium wrapper that stacks k consecutive frames into a single observation.

    This is commonly used in reinforcement learning to provide the agent with
    information about the dynamics of the environment (e.g., velocity).

    Attributes:
        k (int): The number of frames to stack.
        frames (Deque[np.ndarray, None]): A deque to store the most recent k frames.
    """

    def __init__(self, env: gym.Env, k: int) -> None:
        """
        Initializes the FrameStack wrapper.

        Args:
            k (int): The number of frames to stack.

        Raises:
            ValueError: If `k` is not an integer greater than 1, or if the
                        environment's observation space is not defined.
        """
        super().__init__(env)
        if not isinstance(k, int) or k <= 1:
            raise ValueError(f"Frame stack size `k` must be an integer greater than 1, but got {k}.")

        self.k: int = k
        self.frames: Deque[np.ndarray] = deque([], maxlen=k)

        if not self.observation_space or not hasattr(self.observation_space, "shape"):
             raise ValueError("Observation space or its shape is not defined")

        obs_shape: Tuple[int, ...] = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((k,) + obs_shape),
            dtype=self.observation_space.dtype,
        )

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and the frame buffer.

        The frame buffer is populated with k copies of the initial observation.

        Args:
            **kwargs (Any): Keyword arguments for the environment's reset method.

        Returns:
            Tuple[np.ndarray, dict]: The stacked observation and the info dictionary.
        """
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)

        return self._get_obs(), info

    def step(self, action: Any) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Takes a step in the environment and updates the frame buffer.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            Tuple[np.ndarray, SupportsFloat, bool, bool, dict]: A tuple containing
            the stacked observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Retrieves the stacked frames as a single NumPy array.

        Returns:
            np.ndarray: The observation consisting of k stacked frames.
        """
        assert len(self.frames) == self.k, "Frame buffer is not full."
        return np.array(list(self.frames))