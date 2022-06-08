import numpy as np


class Elliptical(object):
    def __init__(self, K, d, s0, l, rng):
        self.K = K
        self.d = d
        self.s0 = s0
        self.rng = rng
        self.mu = np.zeros((K, d))
        self.k = l
        self.A = self.rng.uniform(-1, 1, (d, l))
        self.beta = np.zeros(d)
        self.beta[rng.choice(range(d), s0, replace=False)] = rng.uniform(1, 2, s0)
        self.x = None

    def context(self):
        U = self.rng.randn(self.k, self.K)
        U /= np.linalg.norm(U, axis=0)
        R = self.rng.randn(self.K)
        self.x = self.mu + (R * np.dot(self.A, U)).T
        return self.x

    def pull(self, action):
        err = self.rng.randn()
        ev = np.dot(self.x[action], self.beta)
        reward = ev + err
        regret = np.max(np.dot(self.x, self.beta)) - ev
        return reward, regret

    def false_positive(self, est_beta):
        return np.count_nonzero(est_beta[self.beta == 0])

    def false_negative(self, est_beta):
        return np.count_nonzero(self.beta[est_beta == 0])

    def error_l1(self, est_beta):
        return np.sum(np.abs(self.beta - est_beta))

    def error_l2(self, est_beta):
        return np.linalg.norm(self.beta - est_beta)
