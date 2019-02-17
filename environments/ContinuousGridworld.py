from environments.Environment import Environment
import numpy as np

class CtsGridWorld(Environment):
    _x = 0
    _y = 0

    steps = 0
    noise = 0.01

    def __init__(self, params):
        self.maxSteps = params['steps']
        self.stepSize = params['stepsize']

    def getReward(self):
        if self._x + self.stepSize >= 1 and self._y + self.stepSize >= 1:
            return 1
        return 0

    def step(self, action):
        noise = np.random.normal(0, self.noise)
        if action == 0:
            self._x = bound(self._x + self.stepSize + noise, 0, 1)
        elif action == 1:
            self._y = bound(self._y + self.stepSize + noise, 0, 1)
        elif action == 2:
            self._x = bound(self._x - self.stepSize + noise, 0, 1)
        elif action == 3:
            self._y = bound(self._y - self.stepSize + noise, 0, 1)

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
        return [1, 1]

    def numActions(self):
        return 4

def bound(x, min, max):
    b = max if x >= max else x
    b = 0 if b <= min else b
    return b
