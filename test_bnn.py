import numpy as np
from BNNApproximation import BNNApproximation

def test_bnn():
    bnnApproximation = BNNApproximation(np.array([10, 10]), 4)
    mean = 10
    variance = 2
    np.random.seed(42)
    values = np.random.normal(mean, variance, 100)

    [bnnApproximation.update_stats(np.array([1, 1]), 3, val=x) for x in values]
    samples = np.asarray(bnnApproximation.sample(np.array([1, 1]), 3, 100))
    print("Samples mean", samples.mean())
    print("Samples std dev", samples.std(0, ddof=1))


def main():
    test_bnn()

if __name__ == "__main__": main()
