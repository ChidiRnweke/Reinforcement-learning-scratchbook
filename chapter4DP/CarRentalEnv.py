import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CarRentalEnv(gym.Env):
    def __init__(
        self,
        size=20,
        pricePerCar=10,
        movingCost=2,
        parkingCost=4,
        numFreeParking=10,
        requestsLocationOne=3,
        requestsLocationTwo=4,
        returnsLocationOne=3,
        returnsLocationTwo=2,
    ) -> gym.Env:
        """Initializes Jack's car rental. All the values above are the default from Sutton and Barto's book.

        Args:
            size (int, optional): Amount of cars. Defaults to 20.
            pricePerCar (int, optional): The reward for renting out a car. Defaults to 10.
            movingCost (int, optional): The cost of moving a car from location 1 to 2. Defaults to 2.
            parkingCost (int, optional): The cost of parking cars above numFreeParking. Defaults to 4.
            numFreeParking (int, optional): The amount of cars can be parked for free. Defaults to 10.
            requestsLocationOne (int, optional): Average requests for location 1. Defaults to 3.
            requestsLocationTwo (int, optional): Average requests for location 2. Defaults to 4.
            returnsLocationOne (int, optional): Average returns for location 1. Defaults to 3.
            returnsLocationTwo (int, optional): Average returns for location 2. Defaults to 2.

        Returns:
            gym.Env: Jack's car rental
        """

        self.size = size

        self.observation_space = spaces.Box(low=0, high=size, shape=(2,), dtype=int)

        self.action_space = spaces.Discrete(n=11, start=-5)

        self.pricePerCar = pricePerCar
        self.movingCost = movingCost
        self.parkingCost = parkingCost

        self.numFreeParking = numFreeParking

        self.requestsLocationOne = requestsLocationOne
        self.requestsLocationTwo = requestsLocationTwo

        self.returnsLocationOne = returnsLocationOne
        self.returnsLocationTwo = returnsLocationTwo

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._cars_state = self.np_random.integers(0, self.size, size=2, dtype=int)
        return self._get_obs()

    def _get_obs(self):
        return self._cars_state

    def step(self, action):
        returns_1, returns_2 = self._sampleReturns()
        self._updateState(returns_1, returns_2)
        reward = self._rewardAnd_updateState(action)
        return self._get_obs(), reward, False, None  #

    def _rewardAnd_updateState(self, action: int) -> int:
        """Updates the state and returns a reward

        Args:
            action (int): the action that was taken

        Returns:
            int: the reward
        """
        requests_1, requests_2 = self._sampleRequests()
        income = self._calculateIncome(requests_1, requests_2)
        move_cost = self._calculateMoveCost(action)
        parking_cost = self._calculateParkingCosts(action)
        self._updateState(-requests_1, -requests_2)
        return income - move_cost - parking_cost

    def _updateState(self, param1: int, param2: int) -> None:
        """Updates the state

        Args:
            param1 (int): Returns or requests
            param2 (int): Returns or requests
        """
        self._cars_state[0] = np.clip(self._cars_state[0] + param1, a_min=0, a_max=20)
        self._cars_state[1] = np.clip(self._cars_state[1] + param2, a_min=0, a_max=20)

    def _calculateIncome(self, served1, served2):
        return self.pricePerCar * (served1 + served2)

    def _sampleRequests(self):
        return (
            np.minimum(
                self._cars_state[0], np.random.poisson(lam=self.requestsLocationOne)
            ),
            np.minimum(
                self._cars_state[1], np.random.poisson(lam=self.requestsLocationTwo)
            ),
        )

    def _sampleReturns(self):
        return (
            np.random.poisson(lam=self.returnsLocationOne),
            np.random.poisson(lam=self.returnsLocationTwo),
        )

    def _calculateMoveCost(self, action):
        move_cost = np.abs(action) * self.movingCost
        # Moving one car is free
        if action > 0:
            move_cost -= self.movingCost
        return move_cost

    def _calculateParkingCosts(self, action):
        parking_cost = self.parkingCost * np.maximum(
            0, self._cars_state[0] + action - 10
        )
        parking_cost += self.parkingCost * np.maximum(
            0, self._cars_state[1] - action - 10
        )
        return parking_cost

