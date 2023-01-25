import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom

def graph_binomial_distribution(n, p, c_level):

    # Create k, prob and cumulative_prob
    k = np.arange(0, n+1)
    binomial_prob = binom.pmf(k, n, p)
    cumulative_prob = np.cumsum(binomial_prob)

    # create plot
    plt.bar(k, binomial_prob, color='g')

    plt.xlabel("k")
    plt.ylabel("Probability")
    plt.title(f"Binomial Distribution (n={n}, p={p})")

    # find confidence_level_cutoffs

    left_cutoff_value = np.searchsorted(cumulative_prob, c_level, side="left")
    right_cutoff_value = np.searchsorted(cumulative_prob, 1-c_level, side="left") + 1

    for i in range(left_cutoff_value):
        plt.bar(k[i], binomial_prob[i], color='r')

    for j in range(right_cutoff_value, n+1):
        plt.bar(k[j], binomial_prob[j], color='r')
    # show graph
    plt.show()

