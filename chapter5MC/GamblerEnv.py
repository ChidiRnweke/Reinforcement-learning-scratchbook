import gymnasium as gym
import numpy as np


class GamblerEnv(gym.Env):
    def __init__(self, observationSpace=101, actionSpace=100, ph=0.45) -> gym.Env:

        self.observation_space = gym.spaces.Discrete(n=observationSpace, start=0)

        self.action_space = gym.spaces.Discrete(n=actionSpace, start=0)
        self.ph = ph

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._state = self.observation_space.sample()
        return self._get_obs()

    def _get_obs(self):
        return self._state

    def step(self, action):
        if self.np_random.binomial(n=1, p=self.ph):
            self._state += action
        else:
            self._state -= action

        self._state = np.clip(self._state, a_min=0, a_max=100)

        terminated = self.determineTermination()
        reward = self.calculateReward()
        return self._get_obs(), reward, terminated

    def determineTermination(self):
        if self._state in (0, 100):
            terminated = True
        else:
            terminated = False
        return terminated

    def calculateReward(self) -> int:
        if self._state == 100:
            reward = 1
        else:
            reward = 0
        return reward


class MonteCarloAgent:
    def __init__(self, observationSpace=101, actionSpace=100, discount=1, epsilon=0.1):
        self.observationSpace = np.arange(observationSpace)
        self.actionSpace = np.arange(actionSpace)
        self.epsilon = epsilon
        self.Q = np.random.random(size=(observationSpace, actionSpace))
        self.policy = self._initializePolicy()

        self.episode = []
        self.rewards = []
        self.n = np.ones_like(self.Q)
        self.discount = discount

    def move(self, state):
        action = np.random.choice(self.actionSpace, p=self.policy[state, :])
        self.episode.append((state, action))
        return action

    def observeReward(self, reward):
        self.rewards.append(reward)

    def update(self):
        G = 0
        visited = []
        for stateActionPair, reward in zip(
            reversed(self.episode), reversed(self.rewards)
        ):
            if stateActionPair not in visited:
                state, action = stateActionPair
                if state == 0:
                    continue
                G = (self.discount * G) + reward
                self.n[stateActionPair] += 1
                self.Q[state, action] += (
                    1.0 / self.n[state, action] * (G - self.Q[state, action])
                )
                self.Q[state, state + 1 :] = 0
                bestActions = np.flatnonzero(
                    self.Q[state] == np.max(self.Q[state])
                )  # Ties are possible
                bestAction = (
                    np.random.choice(bestActions)
                    if len(bestActions) > 1
                    else bestActions[0]
                )

                self.policy[state, : state + 1] = self.epsilon / (state + 1)
                self.policy[state, bestAction] = (
                    1 - self.epsilon + (self.epsilon / (state + 1))
                )
                self.policy[state, :] = self.policy[state, :] / np.linalg.norm(
                    self.policy[state, :], ord=1
                )
                visited.append(stateActionPair)

        self.rewards = []
        self.episode = []

    def _initializePolicy(self):
        pol = np.zeros_like(self.Q)
        for state, _ in enumerate(pol):
            if state in (0, 100):
                pol[state, 0] = 1
                continue

            self.Q[state, state + 1 :] = 0
            pol[state, : state + 1] = 1 / (state + 1)
        return pol


class OffPolicyMonteCarloAgent(MonteCarloAgent):
    def __init__(
        self, b, observationSpace=101, actionSpace=100, discount=1, epsilon=0.1,
    ):
        super().__init__(observationSpace, actionSpace, discount, epsilon)
        self.b = b
        self.C = np.zeros_like(self.Q)

    def update(self):
        G = 0
        W = 1
        visited = []
        for stateActionPair, reward in zip(
            reversed(self.episode), reversed(self.rewards)
        ):

            if stateActionPair not in visited:
                state, action = stateActionPair
                if state == 0:
                    continue
                G = (self.discount * G) + reward
                self.C[state, action] += W
                self.Q[state, action] += (W / self.C[state, action]) * (
                    G - self.Q[state, action]
                )
                self.Q[state, state + 1 :] = 0
                bestActions = np.flatnonzero(
                    self.Q[state] == np.max(self.Q[state])
                )  # Ties are possible
                if len(bestActions) == 0:
                    bestAction = np.random.randint(high=state + 1)
                bestAction = (
                    np.random.choice(bestActions)
                    if len(bestActions) > 1
                    else bestActions[0]
                )

                self.policy[state, : state + 1] = self.epsilon / (state + 1)
                self.policy[state, bestAction] = (
                    1 - self.epsilon + (self.epsilon / (state + 1))
                )
                self.policy[state, state + 1 :] = 0

                self.policy[state, :] = self.policy[state, :] / np.linalg.norm(
                    self.policy[state, :], ord=1
                )
                visited.append(stateActionPair)
                self.n[state, action] += 1

                W = W / self.b[state, action]
        self.rewards = []
        self.episode = []
