from agents.Agent import Agent

class Optimal(Agent):
  def __init__(self, num_states, num_acts):
    self.num_states = num_states
    self.num_acts = num_acts

  def policy(self, s):
    return 1
  def getAction(self, s):
    return 1

  def update(self, s, sp, r, a, done):
    return None

  def start(self, s):
    return 1
