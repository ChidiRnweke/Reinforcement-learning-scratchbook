import gymnasium as gym
import numpy as np


class CliffworldEnv(gym.Env):
    def __init__(self, size, cliff=None) -> gym.Env:

        self.observation_space = gym.spaces.Box(
            low=0, high=np.array([size // 3, size]), dtype=np.int16
        )
        self.r = (size // 3) - 1
        self.c =  size - 1
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([1, 1]),
            5: np.array([-1, -1]),
            6: np.array([-1, 1]),
            7: np.array([1, -1]),
            8: np.array([0, 0]),
        }
        if cliff is None:
            self.cliff = np.arange(1, self.c)
            self.cliff = tuple(
                np.stack((np.zeros_like(self.cliff), self.cliff), axis=-1)
            )

    def reset(self, seed=None, default=True):
        super().reset(seed=seed)

        if default:
            self._state = np.array([0, 0])
            self.target = np.array([0, self.c])
        else:
            self._state = self.observation_space.sample()
            self.target = self._state
            while np.array_equal(self.target, self._state):
                self.target = self.observation_space.sample()
        return self._get_obs()

    def _get_obs(self):
        return tuple(self._state)

    def step(self, action):
        direction = self._action_to_direction[action].copy()
        self._state = np.clip(self._state + direction, 0, a_max=[self.r, self.c])
        if (terminated := any([np.array_equal(self._state, cliff) for cliff in self.cliff])) :
            reward = -100
            terminated = True
        elif (terminated := np.array_equal(self._state, self.target)) :
            reward = 0
        else:
            terminated = False
            reward = -1
        return self._get_obs(), reward, terminated

    