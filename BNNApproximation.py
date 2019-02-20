import numpy as np
from BNN import new_bnn
import tensorflow as tf
from bayesianapproximator import BayesianApproximator

class BNNApproximation(BayesianApproximator):

    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        self.session = tf.InteractiveSession()
        self.bnn = new_bnn()
        self.bnn.compile(optimizer=tf.keras.optimizers.SGD(1e-1),
            loss=tf.losses.huber_loss)


    def update_stats(self, s, a, val=0.0, batch_size = 1, epochs = 10):
        if self.dimensions.shape != s.shape or not a <= self.num_actions:
            raise ValueError("Invalid value to update stats", s, a, val)

        # print("State action", s, a)
        input_vector = np.concatenate((self.convert_state(s), self.convert_action(a)))
        # print("The input vector shape", input_vector.shape)
        # print("The input vector", input_vector)


        self.bnn.fit(np.array([[input_vector]]), np.array([[val]]), batch_size=batch_size, epochs=epochs, verbose=0)


    def sample(self, s, a, n):
        if self.dimensions.shape != s.shape or not a <= self.num_actions:
            raise ValueError("Invalid value to sample", s, a)

        input_vector = np.concatenate((self.convert_state(s), self.convert_action(a)))

        sample_fn = self.bnn(np.array([[input_vector]], dtype=np.float32))
        predictions = [self.session.run(sample_fn) for i in range(n)]
        return predictions

    def convert_state(self, s):
        vectors = []
        for i in range(len(self.dimensions)):
            arr = np.zeros(self.dimensions[i])

            arr[s[i]] = 1
            vectors.append(arr)
        return np.concatenate(vectors)

    def convert_action(self, a):
        arr = np.zeros(self.num_actions)
        arr[a-1] = 1
        return arr
