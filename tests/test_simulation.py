from simulation import counts_from_distribution, compute_net
import numpy as np


def test_counts_from_distribution():
    dist = [
        [0 for x in range(6)] + [1],
        [0 for x in range(6)] + [1],
        [0 for x in range(6)] + [1],
        [0 for x in range(6)] + [1],
        [0 for x in range(6)] + [1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ]
    counts = [0, 0, 0, 0, 0, 10, 5]
    res = counts_from_distribution(counts, dist)
    assert res == [0, 5, 0, 0, 10, 0, 0]


def test_compute_net():
    a = [5, 10, 5, 5, 10, 15, 20]
    b = [0, 5, 5, 5, 15, 10, 30]
    counts = [a, b]
    res = compute_net(counts)
    a_net = (sum(a[5:]) - sum(a[:4])) / sum(a)
    b_net = (sum(b[5:]) - sum(b[:4])) / sum(b)
    exp = np.mean([a_net, b_net])

    assert res == exp
