import tensorflow as tf
import numpy as np

class LinearModel(object):
    def __init__(self, weights):
        self.weights = [weights]
        print(weights.shape)

    def __call__(self, phi):
        return phi @ self.weights[0]

    def predict(self, phi):
        phi = tf.convert_to_tensor(phi, dtype=np.float32)
        return self(phi)
