import numpy as np


class QlearningAgent:
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
            R + (self.g * np.max(self.Q[Sp]) - self.Q[S][A])
        )

    def policyImprovement(self, S):
        actions = np.flatnonzero(self.Q[S] == np.max(self.Q[S]))
        SA = np.random.choice(actions) if len(actions) > 1 else actions.item()
        policy = np.ones(self.actions) * self.epsilon / self.actions
        policy[SA] = 1 - self.epsilon + (self.epsilon / self.actions)
        return policy


class DoubleQLearningAgent:
    def __init__(
        self, epsilon=0.1, lr=0.5, stateSpace=(7, 10), actionSpace=4, g=1
    ) -> None:
        self.epsilon = epsilon
        self.lr = lr
        self.g = g
        self.actions = actionSpace
        self.Q1 = np.zeros(shape=(stateSpace[0], stateSpace[1], actionSpace))
        self.Q2 = np.zeros(shape=(stateSpace[0], stateSpace[1], actionSpace))
        self.Q = np.zeros(shape=(stateSpace[0], stateSpace[1], actionSpace))
        self.policy = np.ones_like(self.Q) / actionSpace

    def move(self, state):
        return np.random.choice(self.actions, p=self.policy[state])

    def update(self, S, A, R, Sp, Ap):
        if np.random.rand > 0.5:
            self.Q1[S][A] = self.valueEvaluation(S, A, R, Sp, Ap, self.Q1, self.Q2)
        else:
            self.Q2[S][A] = self.valueEvaluation(S, A, R, Sp, Ap, self.Q2, self.Q1)
        self.Q = self.Q1 + self.Q2
        self.policy[S] = self.policyImprovement(S)

    def valueEvaluation(self, S, A, R, Sp, Q1, Q2):
        return Q1[S][A] + self.lr * (
            R + (self.g * Q2[Sp][np.argmax(Q1[Sp])] - self.Q[S][A])
        )

    def policyImprovement(self, S):
        actions = np.flatnonzero((self.Q[S]) == np.max(self.Q[S]))
        SA = np.random.choice(actions) if len(actions) > 1 else actions.item()
        policy = np.ones(self.actions) * self.epsilon / self.actions
        policy[SA] = 1 - self.epsilon + (self.epsilon / self.actions)
        return policy
