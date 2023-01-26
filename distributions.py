import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom
import math


class BinomialDistribution:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.mean = self.n * self.p
        self.sigma = math.sqrt(self.mean * (1 - self.p))
        self.variance = self.sigma ** 2

        # create np arrays of values

        self.range = np.arange(0, self.n + 1)
        self.probabilities = binom.pmf(self.range, self.n, self.p)
        self.cumulative_probabilities = np.cumsum(self.probabilities)
        self.z_scores = (self.range - self.mean) / self.sigma

    def binomial_prob(self, k):
        return binom.pmf(k, self.n, self.p)

    def cumulative_prob(self, k):
        return binom.cdf(k, self.n, self.p)

    def find_c_level(self, k):
        if k < self.mean:
            return binom.cdf(k, self.n, self.p)
        elif k > self.mean:
            return binom.cdf(k, self.n, self.p)

    def cutoff_zscore(self, z_score):
        return np.searchsorted(self.z_scores, z_score)

    def cutoff_confidence_level(self, c_level, side='left'):
        if side == 'left':
            return np.searchsorted(self.cumulative_probabilities, c_level, side='left') - 1
        elif side == 'right':
            return np.searchsorted(self.cumulative_probabilities, 1 - c_level, side="left") + 1
        else:
            raise ValueError("left or right expected")

    def graph(self, *, c_level=None, z=None):

        # create bar graph

        plt.bar(self.range, self.probabilities, color='g')

        plt.xlabel("k")
        plt.ylabel("Probability")
        plt.title(f"Binomial Distribution (n={self.n}, p={self.p})")

        if c_level and z:
            raise Exception("graph() takes only either c_level or z_score not both")

        if c_level:
            for i in range(self.cutoff_confidence_level(c_level) + 1):
                plt.bar(self.range[i], self.probabilities[i], color='r')
            for j in range(self.cutoff_confidence_level(c_level, side='right'), self.n + 1):
                plt.bar(self.range[j], self.probabilities[j], color='r')

        if z:
            for i in range(self.cutoff_zscore(-z)):
                plt.bar(self.range[i], self.probabilities[i], color='r')
            for j in range(self.cutoff_zscore(z), self.n + 1):
                plt.bar(self.range[j], self.probabilities[j], color='r')
        # show graph

        plt.show()
