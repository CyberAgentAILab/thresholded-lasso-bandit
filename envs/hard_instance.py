import numpy as np


class HardInstance(object):
    def __init__(self, d, rho_sq, rng):
        self.K = 3
        self.d = d
        self.rng = rng
        self.beta = np.concatenate([[1, 0.1, 1], np.zeros(d - 3)])
        self.V = (1 - rho_sq) * np.eye(self.K) + rho_sq * np.ones((self.K, self.K))
        self.support_xs = [
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0.9, 0.5, 0]]),
            np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0.0, 0.5, 0.9]])
        ]
        self.x = None

    def context(self):
        if self.rng.rand() < 0.3:
            support_x = self.support_xs[0]
        else:
            support_x = self.support_xs[1]
        non_support_x = self.rng.multivariate_normal(np.zeros(self.K), self.V, self.d - 3).T
        self.x = np.concatenate([support_x, non_support_x], axis=1)
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
