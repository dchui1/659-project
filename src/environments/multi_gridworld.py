from src.environments.Environment import Environment
import numpy as np
from src.environments.Renderable import Renderable
class GridWorld(Environment, Renderable):
    _x = 0
    _y = 0


    # AGENT_COLOR = np.array([1, 0, 0])
    # GOAL_COLOR = np.array([0, 1, 0])
    steps = 0

    def __init__(self, params):
        Renderable.__init__(self)
        self.shape = params['shape']
        self.maxSteps = params['steps']
        self.small_reward = params['small_reward']
        self.big_reward = params['big_reward']

    def getReward(self):
        if self._x == self.shape[0] - 1 and self._y == self.shape[1] - 1:
            return self.big_reward
        if self.x == self.shape[0] -1 or self._y == self.shape[1] -1:
            return self.small_reward
        return 0

    def start(self):
        self._x = 0
        self._y = 0
        self.steps = 0
        return np.array([self._x, self._y])

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


        return (r, np.array([self._x, self._y]), done)

    def observationShape(self):
        # cast to list just in case we received a tuple or a np.array
        return list(self.shape)

    def numActions(self):
        return 4


def bound(x, min, max):
    b = max if x >= max else x
    b = 0 if b <= min else b
    return b
