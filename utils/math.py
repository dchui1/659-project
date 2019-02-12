import numpy as np

# takes an array of values and returns the argMax
# the difference between this and np.argmax is that
# this implementation randomly breaks ties
def argMax(arr):
    indices = np.where(arr == np.max(arr))[0]
    if len(indices) < 1:
        raise ArithmeticError()

    return np.random.choice(indices)
