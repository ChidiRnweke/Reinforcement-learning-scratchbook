import numpy as np


class SarsaAgent:
    def __init__(
        self, epsilon=0.1, lr=0.5, stateSpace=(7, 10), actionSpace=4, g=1
    ) -> None:
        self.epsilon = epsilon
        self.lr = lr
        self.g = g
        self.actions = actionSpace
        self.Q = np.zeros(shape=(stateSpace[0], stateSpace[1], actionSpace))
        self.policy = np.ones_like(self.Q) / actionSpace

    def move(self, state):
        return np.random.choice(self.actions, p=self.policy[state])

    def update(self, S, A, R, Sp, Ap):
        self.Q[S][A] = self.valueEvaluation(S, A, R, Sp, Ap)
        self.policy[S] = self.policyImprovement(S)

    def valueEvaluation(self, S, A, R, Sp, Ap):
        return self.Q[S][A] + self.lr * (R + (self.g * self.Q[Sp][Ap]) - self.Q[S][A])

    def policyImprovement(self, S):
        actions = np.flatnonzero(self.Q[S] == np.max(self.Q[S]))
        SA = np.random.choice(actions) if len(actions) > 1 else actions.item()
        policy = np.ones(self.actions) * self.epsilon / self.actions
        policy[SA] = 1 - self.epsilon + (self.epsilon / self.actions)
        return policy


class ExpectedSarsaAgent:
    def __init__(
        self, epsilon=0.1, lr=0.5, stateSpace=(7, 10), actionSpace=4, g=1
    ) -> None:
        self.epsilon = epsilon
        self.lr = lr
        self.g = g
        self.actions = actionSpace
        self.Q = np.zeros(shape=(stateSpace[0], stateSpace[1], actionSpace))
        self.policy = np.ones_like(self.Q) / actionSpace

    def move(self, state):
        return np.random.choice(self.actions, p=self.policy[state])

    def update(self, S, A, R, Sp, Ap):
        self.Q[S][A] = self.valueEvaluation(S, A, R, Sp, Ap)
        self.policy[S] = self.policyImprovement(S)

    def valueEvaluation(self, S, A, R, Sp, Ap):
        return self.Q[S][A] + self.lr * (
            R + (self.g * np.dot(self.Q[Sp], self.policy[Sp])) - self.Q[S][A]
        )

    def policyImprovement(self, S):
        actions = np.flatnonzero(self.Q[S] == np.max(self.Q[S]))
        SA = np.random.choice(actions) if len(actions) > 1 else actions.item()
        policy = np.ones(self.actions) * self.epsilon / self.actions
        policy[SA] = 1 - self.epsilon + (self.epsilon / self.actions)
        return policy
