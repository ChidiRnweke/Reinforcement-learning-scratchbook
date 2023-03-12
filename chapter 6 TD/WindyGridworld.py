import gymnasium as gym
import numpy as np


class WindyGridworldEnv(gym.Env):
    def __init__(self, rows=7, columns=10) -> gym.Env:

        self.observation_space = gym.spaces.Box(
            low=0, high=np.array([rows, columns]), dtype=np.int16
        )
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.wind = np.zeros((rows, columns))
        self.wind[:, 3:9] += 1
        self.wind[:, 6:8] += 1

    def reset(self, seed=None, default=True):
        super().reset(seed=seed)

        if default:
            self._state = np.array([3, 0])
            self.target = np.array([3, 7])
        else:
            self._state = self.observation_space.sample()
            self.target = self._state
            while np.array_equal(self.target, self._state):
                self.target = self.observation_space.sample()
        return self._get_obs()

    def _get_obs(self):
        return self._state

    def step(self, action):
        direction = self._action_to_direction[action]
        direction = direction + self.wind[self._state]
        self._state = np.clip(self._state + direction, 0, self.size - 1)
        terminated = np.array_equal(self._state, self.target)
        reward = 1 if terminated else 0
        return self._get_obs(), reward, terminated
