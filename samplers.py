import numpy as np

class ThompsonSampling(object):
    def __init__(self, n, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        n      (int): number of arms in multiarmed bandit.
        """
        self.n = n
        self._as = [init_a] * self.n
        self._bs = [init_b] * self.n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.n)]

    def get_choice(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.n)]
        i = max(range(self.n), key=lambda x: samples[x])
        return i

    def update(self, choice, didSolve):
        """
        didSolve (bool): whether or not previous choice
        """
        r = int(didSolve)
        self._as[choice] += r
        self._bs[choice] += (1 - r)
