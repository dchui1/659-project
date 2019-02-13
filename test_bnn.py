import numpy as np
from BNNApproximation import BNNApproximation

def test_bnn():
    bnnApproximation = BNNApproximation(np.array([10, 10]), 4)

    values = np.random.normal(10, 2, 1000)
    # sample = bnnApproximation.sample(np.array([1, 1]), 3, 10)

    [bnnApproximation.update_stats(np.array([1, 1]), 3, val=x) for x in values]
    samples = bnnApproximation.sample(np.array([1, 1]), 3, 10)
    print("Sample", samples)


def main():
    test_bnn()

if __name__ == "__main__": main()
