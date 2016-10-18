import numpy as np


def counts_from_distribution(counts, dist):
    total = [0 for x in range(7)]
    for i, c in enumerate(counts):
        obs = np.random.choice(7, c, p=dist[i])
        for x in range(7):
            total[x] = total[x] + sum(obs == x)
    return total


def compute_net(counts):
    tot_counts = np.sum(counts, axis=0)
    return (sum(tot_counts[5:]) - sum(tot_counts[:4]))/sum(tot_counts)
