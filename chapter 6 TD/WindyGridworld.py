import gymnasium as gym
import numpy as np


class WindyGridworldEnv(gym.Env):
    def __init__(self, rows=7, columns=10, stochastic=False) -> gym.Env:

        self.observation_space = gym.spaces.Box(
            low=0, high=np.array([rows, columns]), dtype=np.int16
        )
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
        self.r = rows - 1
        self.c = columns - 1
        self.wind = np.zeros((rows, columns), dtype=np.int64)
        self.wind[:, 3:9] += 1
        self.wind[:, 6:8] += 1
        self.stochastic = stochastic

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
        return tuple(self._state)

    def step(self, action):
        direction = self._action_to_direction[action].copy()
        direction[0] += self.wind[tuple(self._state)]
        if self.stochastic and np.isin(self._state[1], np.arange(3, 9)):
            direction[0] += np.random.choice([-1, 0, 1])
        self._state = np.clip(self._state + direction, 0, a_max=[self.r, self.c])
        terminated = np.array_equal(self._state, self.target)
        reward = 0 if terminated else -1
        return self._get_obs(), reward, terminated
