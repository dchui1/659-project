# -*- coding: utf-8 -*-
"""Copy of Q_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rKKgtfD69kIQbiw_P5RU-snr8sAEWY4G

####Imports and Definitions
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from bayesianapproximator import *

from environments.gridworld import GridWorld

class Optimal:
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


# class Agent:
#     def __init__(self):
#         self.B = None
#
#     def start(self, obs):
#         self.next_action = self.policy(obs)
#         return self.next_action

class Q:
  def __init__(self, num_states, num_acts):
    self.alpha = 0.01
    self.gamma = 0.99
    self.epsilon = 0.1

    self.num_states = num_states
    self.num_acts = num_acts

    self.Q = np.zeros(self.num_states + [self.num_acts])
    self.next_action = 0

  def policy(self, S):
    if random.random() < self.epsilon:
      return random.randint(0, self.num_acts - 1)
    return self.maxAction(S)

  def maxAction(self, s):
    act_vals = self.Q[s, :]
    move = self.breakTie(act_vals)
    return move

  def getAction(self, Obs):
    return self.next_action

  # if gamma_tp1 = 0, that means the episode terminated
  def learn(self, s, sp, r, a, gamma, max_bonus=0):
    ap = self.maxAction(sp)
    Q_p = self.Q[sp, ap]

    tde = (r + max_bonus + gamma * Q_p) - self.Q[s, a]  # add a max_bonus i
    self.Q[s, a] = self.Q[s, a] + self.alpha*tde

  def update(self, S, Sp, r, a, done, max_bonus=0):
    if done:
      self.learn(S, Sp, r, a, 0) # If done, whould we give an exploration bonus?
    else:
      self.next_action = self.policy(Sp)
      self.learn(S, Sp, r, a, self.gamma, max_bonus)



  def breakTie(self, act_vals):
    indexes = np.where(act_vals == np.max(act_vals))[0]
    if len(indexes) < 1:
      raise ArithmeticError()

    return np.random.choice(indexes)

"""####Q-learning Agent with No Bonus"""


"""####Q-learning Agent with Bonus updated Tabularly"""

class QRewardValueFunction(Q):

  def __init__(self, num_states, num_acts, bayesianApproximator):
    super().__init__(num_states, num_acts)
    self.bayesianApproximator = bayesianApproximator
    self.epsilon = 0.01

  def update(self, s, sp, r, a, done):
    self.bayesianApproximator.update_stats(s, a, r)
    bonus = max(self.bayesianApproximator.sample(s, a, 10))
    super().update(s, sp, r + bonus, a, done)

  def start(self, obs):
      self.next_action = self.policy(obs)
      return self.next_action


def runExperiment(env, num_episodes, q):
  total_reward = 0
  rewards = []
  steps = []

  for episode in range(num_episodes):

    s = env.reset()
    a = q.start(s)
    done = False

    step = 0

    while not done:
      (sp, r, done, __) = env.step(a) # Note: the environment "registers" the new sp as env.pos
      q.update(s, sp, r, a, done)

      s = sp # update the current state to sp
      a = q.getAction(s) # update the current action to a

      total_reward += r
      rewards.append(total_reward)

      step += 1

    steps.append(step)

  return (steps, rewards)


def averageOverRuns(Agent, env, runs = 20):
  rewards = []
  total_steps = []
  for run in range(runs):
    np.random.seed(run)
    random.seed(run)
    bayesianApproximator = TabularBayesianApproximation(env.observationShape(), env.numActions())
    agent = Agent(env.observationShape(), env.numActions(), bayesianApproximator)
    (steps, r) = runExperiment(env, 500, agent)
    rewards.append(r)
    total_steps.append(steps)


  print("Values of table afterwards: ",  bayesianApproximator.B)

  metric = np.array(total_steps)
  mean = metric.mean(axis=0)
  stderr = metric.std(axis=0) / np.sqrt(runs)

  return (mean, stderr)

def confidenceInterval(mean, stderr):
  return (mean - stderr, mean + stderr)

def plotRewards(ax, rewards, stderr, label):
  (low_ci, high_ci) = confidenceInterval(rewards, stderr)
  ax.plot(rewards, label=label)
  ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)

# fig = plt.figure()
# ax = plt.axes()

# def main():
env = GridWorld([30, 30], 400)

# Optimal for riverswim, doesn't make sense on gridworld
# (rewards, stderr) = averageOverRuns(Optimal, env, 20)
# plotRewards(ax, rewards, stderr, 'Optimal')
#
# (rewards, stderr) = averageOverRuns(Q, env, 1)
# plotRewards(ax, rewards, stderr, 'Q epsilon=0.1')

(rewards, stderr) = averageOverRuns(QRewardValueFunction, env, 1)
# plotRewards(ax, rewards, stderr, 'QReward value-function')

# plt.legend()
# plt.show()



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--environment", type=str, default="gridworld", help="environment")
    parser.add_argument("--agent", type=str, default="Q", help="environment")


    return parser.parse_args()

# Notes:
# windy gridworld -> stochastic world.. maybe ignore stochasticity at first
# Try mountain car? This is a continuous-state domain
# river swim: states have far enough variance... How is this determined?
