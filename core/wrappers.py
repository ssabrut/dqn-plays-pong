import cv2
import gymnasium as gym
import numpy as np
from typing import Any

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space must be of type Box"
        
        obs_shape = env.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: Any) -> Any:
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)