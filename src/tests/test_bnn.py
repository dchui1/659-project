import numpy as np
from src.bayesian_inference.BNNApproximation import *
import math
import pytest


@pytest.mark.skip(reason="initialization was refactored, need to update")
def test_bnn():
    # values = np.zeros(1000)

    # get_z_score(10, values, 0, 0.1)
    z_score_limit = 2.
    mu = 10.0
    sigma = 1.0
    np.random.seed(42)
    values = np.random.normal(mu, sigma, 1000)

    empirical_mean = values.mean()
    empirical_stddev = np.std(values)
    print("Empirical mean", empirical_mean)
    print("Empirical std dev", empirical_stddev)
    # get_z_score(100, values, empirical_mean, empirical_stddev)
    get_z_score(10, values, empirical_mean, empirical_stddev)



def get_z_score(epochs, values, empirical_mean, empirical_stddev):
    dimensions = np.array([2, 2])
    num_actions = 4
    bnnApproximation = BNNApproximation(dimensions, num_actions, -1, 0.5, 0.1, 0.1)
    x = np.concatenate((convert_state(np.array([1, 1]), dimensions), convert_action(3, num_actions)))
    print("X", x)
    [bnnApproximation.update_stats(x, val=y, epochs=epochs) for y in values]
    samples = np.asarray(bnnApproximation.sample(x, 100))
    # print("The samples", samples)
    print("Sampled mean", samples.mean())
    print("Sampled std dev", samples.std())
    se = empirical_stddev / math.sqrt(len(samples))
    z_score = (samples.mean() - empirical_mean) / se

    print("Z score", z_score)


def convert_state(s, dimensions):
    sparse_vectors = []
    for i in range(len(dimensions)):
        arr = np.zeros(dimensions[i])
        arr[s[i]] = 1
        sparse_vectors.append(arr)
    return np.concatenate(sparse_vectors)

def convert_action(a, num_actions):
    arr = np.zeros(num_actions)
    arr[a] = 1
    return arr


def main():
    test_bnn()

if __name__ == "__main__": main()
