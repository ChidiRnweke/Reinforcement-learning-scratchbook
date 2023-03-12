import numpy as np


class SarsaAgent:
    def __init__(
        self, epsilon=0.1, lr=0.5, stateSpace=(7, 10), actionSpace=4, g=1
    ) -> None:
        self.epsilon = epsilon
        self.lr = lr
        self.g = g
        self.actions = actionSpace
        self.Q = np.zeros(shape=(stateSpace, actionSpace))
        self.policy = np.ones_like(self.Q) / actionSpace

    def move(self, state):
        return np.random.choice(self.actions, p=self.policy[state, :])

    def update(self, S, A, R, Sp, Ap):
        self.Q[S, A] = self.valueEvaluation(S, A, R, Sp, Ap)
        self.policy[S] = self.policyImprovement(S)

    def valueEvaluation(self, S, A, R, Sp, Ap):
        return self.Q[S, A] + self.lr(R + (self.g * self.Q[Sp, Ap]) - self.Q[S, A])

    def policyImprovement(self, S):
        actions = np.flatnonzero(self.Q[S] == np.max(self.Q[S]))
        policy = np.empty_like(self.policy)
        SA = np.random.choice(actions) if len(actions) > 1 else actions.item()
        policy[S, :] = self.epsilon / self.actions
        policy[S, SA] = 1 - self.epsilon + (self.epsilon / S)
        return policy
