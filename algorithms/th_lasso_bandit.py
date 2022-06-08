import numpy as np

from sklearn.linear_model import Lasso, LinearRegression


def linear_regression(x, y):
    try:
        CXX = np.dot(x.T, x) / x.shape[0]
        CXY = np.dot(x.T, y) / x.shape[0]
        return np.linalg.solve(CXX, CXY).T
    except np.linalg.LinAlgError:
        return LinearRegression(fit_intercept=False).fit(x, y).coef_


class THLassoBandit(object):
    def __init__(self, rng, K, d, lam0):
        self.rng = rng
        self.K = K
        self.d = d
        self.lam0 = lam0
        self.beta = np.zeros(d)
        self.xs = []
        self.rs = []
        self.S = np.arange(d)

    def choose_action(self, x, t):
        if len(self.S) == 0:
            a = self.rng.randint(self.K, dtype=np.int64)
        else:
            a = np.argmax(np.dot(x, self.beta))
        self.xs.append(x[a].copy())
        return a

    def update_beta(self, r, t):
        self.rs.append(r)
        lam = self.lam0 * np.sqrt(2 * np.log(t) * np.log(self.d) / t)
        beta = Lasso(alpha=lam, fit_intercept=False).fit(self.xs, self.rs).coef_
        self.S = np.where(np.abs(beta) > 4 * lam)[0]
        if len(self.S) == 0:
            self.beta = np.zeros(self.d)
            return
        for i in range(1):
            beta_cp = np.zeros(self.d)
            beta_cp[self.S] = beta[self.S]
            self.S = np.where(np.abs(beta_cp) > 4 * lam * np.sqrt(len(self.S)))[0]
            if len(self.S) == 0:
                self.beta = np.zeros(self.d)
                return
            beta[self.S] = linear_regression(np.array(self.xs)[:, self.S], self.rs)
        self.beta = np.zeros(self.d)
        self.beta[self.S] = beta[self.S]
