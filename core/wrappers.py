from typing import Any, Union

import cv2
import gymnasium as gym
import numpy as np


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        if self.observation_space and self.observation_space.shape:
            obs_shape = self.observation_space.shape[:2]
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )
        else:
            raise ValueError("Observation space shape is not defined")

    def observation(self, observation: Any) -> Any:
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape: Union[int, tuple]) -> None:
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        if self.observation_space and self.observation_space.shape:
            obs_shape = self.shape + self.observation_space.shape[2:]
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )
        else:
            raise ValueError("Observation space shape is not defined")

    def observation(self, observation: Any) -> Any:
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
