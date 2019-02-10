from environments.Environment import Environment
import numpy as np

class GridWorld(Environment):
    _x = 0
    _y = 0

    steps = 0

    def __init__(self, shape, maxSteps):
        self.shape = shape
        self.maxSteps = maxSteps

    def getReward(self):
        if self._x == self.shape[0] - 1 and self._y == self.shape[1] - 1:
            return 1
        return 0

    def step(self, action):
        if action == 0:
            self._x = bound(self._x + 1, 0, self.shape[0] - 1)
        elif action == 1:
            self._y = bound(self._y + 1, 0, self.shape[1] - 1)
        elif action == 2:
            self._x = bound(self._x - 1, 0, self.shape[0] - 1)
        elif action == 3:
            self._y = bound(self._y -1, 0, self.shape[1] - 1)

        r = self.getReward()
        self.steps += 1
        done = r == 1 or self.maxSteps == self.steps


        return (np.array([self._x, self._y]), r, done, action)

    def reset(self):
        self.steps = 0
        self._x = 0
        self._y = 0

        return np.array([0, 0])

    def observationShape(self):
        # cast to list just in case we received a tuple or a np.array
        return list(self.shape)

    def numActions(self):
        return 4

def bound(x, min, max):
    b = max if x >= max else x
    b = 0 if b <= min else b
    return b
