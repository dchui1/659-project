import numpy as np
from environments.Environment import Environment
import random

class RiverSwim(Environment):
  steps = 0
  def __init__(self):
    self.STEPS_LIMIT = 10000 # number of steps in each episode?
    self.pos = 0
    self.swimRightStay = 0.6
    self.swimRightUp = 0.35
    self.swimRightDown = 0.05
    self.S1swimRightUp = 0.6
    self.SNswimRightDown = 0.4

  def reset(self):
      self.n = 0
      self.pos = 0
      return self.pos

  def step(self, a): # the transition function?
    old_pos = self.pos
    # if action is 0 then we do nothing
    if a == 1:
      # determine if we will successfully take the "up" action
      flip = random.random()
      if self.pos <= 0: # first state in chain
        if flip > self.S1swimRightUp:
          self.pos = self.pos + 1
      elif self.pos >= 5: # end of chain
        if flip <= self.SNswimRightDown:
          self.pos = self.pos - 1
      else: # middle of chain
        if flip <= self.swimRightDown:
          self.pos = self.pos - 1
        elif flip > self.swimRightDown + self.swimRightStay:
          self.pos = self.pos + 1
    # make sure that the position we return (the next state) is between 0 and 5
    self.pos = np.clip(self.pos, 0, 5)

    done = self.STEPS_LIMIT == self.steps
    self.steps += 1

    # tuple indicating (state, reward, terminated, action)
    # note this is a continuing task, so the environment will only
    # terminate when max number of steps is reached
    return (self.pos, self.rewardFunction(old_pos, a), done, a)

  def rewardFunction(self, x, a):
    if x >= 5 and a == 1:
      return 1.0
    if x <= 0 and a == 0:
      return 5.0/1000.0
    return 0.0

  def observationShape(self):
    # position on the river.
    # states are: 0, 1, 2, 3, 4, 5
    return [6]

  def numActions(self):
    # (0) stay or (1) swim up the river
    return 2
