import numpy as np
from BNNApproximation import *
import math


def test_bnn():
    z_score_limit = 2.
    mu = 10.0
    sigma = 1.0
    np.random.seed(42)
    values = np.random.normal(mu, sigma, 1000)

    empirical_mean = values.mean()
    empirical_stddev = np.std(values)
    print("Empirical mean", empirical_mean)
    print("Empirical std dev", empirical_stddev)
    get_z_score(100, values, empirical_mean, empirical_stddev)
    get_z_score(10, values, empirical_mean, empirical_stddev)



def get_z_score(epochs, values, empirical_mean, empirical_stddev):
    bnnApproximation = BNNApproximation(np.array([10, 10]), 4)
    [bnnApproximation.update_stats(np.array([1, 1]), 3, val=x, epochs=epochs) for x in values]
    samples = np.asarray(bnnApproximation.sample(np.array([1, 1]), 3, 100))
    print("Samples mean", samples.mean())
    print("Samples std dev", samples.std())
    se = empirical_stddev / math.sqrt(len(samples))
    z_score = (samples.mean() - empirical_mean) / se

    print("Z score", z_score)



    # new_samples = np.asarray(bnnApproximation.sample(np.array([10, 10]), 1, 100))
    # print("New Samples mean", new_samples.mean())
    # print("New Samples std dev", new_samples.std())


def main():
    test_bnn()

if __name__ == "__main__": main()
