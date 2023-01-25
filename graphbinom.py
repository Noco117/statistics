import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom
import math

def graph_binomial_distribution(n, p, c_level=0.1, z=None):

    # Create k, prob and cumulative_prob
    k = np.arange(0, n+1)
    binomial_prob = binom.pmf(k, n, p)
    cumulative_prob = np.cumsum(binomial_prob)
    mean = n * p
    sigma = math.sqrt(n * p * (1-p))
    z_scores = (k-mean)/sigma

    # create plot
    plt.bar(k, binomial_prob, color='g')

    plt.xlabel("k")
    plt.ylabel("Probability")
    plt.title(f"Binomial Distribution (n={n}, p={p})")

    # find confidence_level_cutoffs

    left_cutoff_value = (np.searchsorted(cumulative_prob, c_level, side="left") if z is None else np.searchsorted(z_scores, -z))
    right_cutoff_value = (np.searchsorted(cumulative_prob, 1-c_level, side="left") + 1 if z is None else np.searchsorted(z_scores, z))

    for i in range(left_cutoff_value):
        plt.bar(k[i], binomial_prob[i], color='r')

    for j in range(right_cutoff_value, n+1):
        plt.bar(k[j], binomial_prob[j], color='r')
    # show graph
    plt.show()

