import numpy as np
from BNN import new_bnn
import tensorflow as tf
from bayesianapproximator import BayesianApproximator

class BNNApproximation(BayesianApproximator):

    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        self.session = tf.InteractiveSession()
        self.bnn = new_bnn()
        #TODO refactor this back out into the BNN method
        self.bnn.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
            loss=tf.losses.huber_loss)
        # input = np.array(state_dimensions + self.convert_action(self.num_actions))
        # self.history = self.bnn.fit(input, 0, batch_size=32,
        #     epochs=1000, verbose=1)


    def update_stats(self, x, val=0.0, batch_size = 32, epochs = 100):
        s = x[:-1]
        a = x[len(x) - 1]
        if self.dimensions.shape != s.shape or not a <= self.num_actions:
            raise ValueError("Invalid value to update stats", s, a, val)

        input_vector = np.concatenate((s, self.convert_action(a)))
        print(input_vector.shape)

        self.bnn.fit(np.array([[input_vector]]), np.array([[val]]), batch_size=batch_size, epochs=epochs)


    def sample(self, x, n):
        s = x[:-1]
        a = x[len(x) - 1]
        if self.dimensions.shape != s.shape or not a <= self.num_actions:
            raise ValueError("Invalid value to sample", s, a)

        input_vector = np.concatenate((s, self.convert_action(a)))

        # with self.session as sess:
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.tables_initializer())
        sample_fn = self.bnn(np.array([[input_vector]], dtype=np.float32))
        predictions = [self.session.run(sample_fn) for i in range(n)]
        return predictions

    def convert_action(self, a):
        arr = np.zeros(self.num_actions)
        arr[a-1] = 1
        return arr
