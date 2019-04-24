# -*- coding: utf-8 -*-
"""Copy of Q_learning.ipynb
Automatically generated by Colaboratory.
Original file is located at
   https://colab.research.google.com/drive/1rKKgtfD69kIQbiw_P5RU-snr8sAEWY4G
####Imports and Definitions
"""
import random
import math
import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as pet
from utils.bayesianapproximator import *
from utils.BNNApproximation import BNNApproximation
from utils.ExperimentDescription import ExperimentDescription
import utils.registry as registry
from pickle import dump
from agents.TabularQ import TabularQ
import time
from time import sleep


def runExperiment(env, num_episodes, agent, render):
   total_reward = 0
   rewards = []
   steps = []

   # For debugging: store action values and max variance
   num_states = np.prod(env.observationShape())
   num_acts = env.numActions()
   # ac_vals_list = []
   agent.rewardApprox.nu_0  # prior "observations that make the prior mean"
   agent.rewardApprox.alpha_0  # prior IG shape
   agent.rewardApprox.beta_0  # prior IG scale
   prior_var = agent.rewardApprox.beta_0 * (agent.rewardApprox.nu_0 + 1) / (
       agent.rewardApprox.alpha_0 * agent.rewardApprox.nu_0)
   t_dist_vars = []
   bonuses = []
   for j in range(num_acts):
       l_a = []
       b_a = []
       for i in range(num_states):
           l_s = []
           l_s.append(prior_var)
           l_a.append(l_s)
           b_s = []
           b_a.append(b_s)
       t_dist_vars.append(l_a)
       bonuses.append(b_a)
   max_var_T = []
   max_bonus = []

   for episode in range(num_episodes):
       s = env.reset()
       a = agent.start(s)
       if episode == 0:
           print("initial action = ", a)
       done = False
       step = 0

       while not done:
           if render:
               # print("Render env")
               env.render()
               time.sleep(0.07)

           (sp, r, done, __) = env.step(a)  # Note: the environment "registers" the new sp as env.pos
           agent.update(s, sp, r, a, done)

           # For debugging: store the t-distribution variance at each timestep:
           # mu, nu, alpha, beta = self.B[x, :]
           s_idx = np.ravel_multi_index(s, agent.state_shape)
           if s_idx == 0:
               x = agent.getIndex(s) + (a * num_states)
               nu = agent.rewardApprox.B[x, 1]
               alpha = agent.rewardApprox.B[x, 2]
               beta = agent.rewardApprox.B[x, 3]
               t_distribution_var = beta * (nu + 1) / (alpha * nu)
               t_dist_vars[a][s_idx].append(t_distribution_var)
               bonus_t = agent.rewardApprox.bonus
               bonuses[a][s_idx].append(bonus_t)
               # print(bonuses[a][s_idx])
           # ac_vals_list.append(agent.act_vals)

           # For debugging: store the maximum variance for each timestep:
           # var_single_timestep = []
           # bonus_single_timestep = []
           # for x_i in range(num_states * num_acts):
           #     nu_i = agent.rewardApprox.B[x_i, 1]
           #     alpha_i = agent.rewardApprox.B[x_i, 2]
           #     beta_i = agent.rewardApprox.B[x_i, 3]
           #     t_distribution_var = beta_i * (nu_i + 1) / (nu_i * alpha_i)
           #     var_single_timestep.append(t_distribution_var)
           #     bonus_single_timestep.append(agent.rewardApprox.bonus)
           # max_var_T.append(np.max(var_single_timestep))
           # max_bonus.append(np.max(bonus_single_timestep))
           # data_dict_max_var = {'max_var_T_array': max_var_T,
           #                      'max_bonus': max_bonus
           #     }

           s = sp
           a = agent.getAction(s)
           # print(agent.act_vals)
           total_reward += r
           rewards.append(total_reward)
           step += 1
       steps.append(step)
       print("Episode", episode, " Step", step)

   data_dict = {
           't_dist_var_along_trajectory': t_dist_vars,
           'all_bonuses': bonuses
       }
   # np.save("tmp/TabR_TabQ_maxvars", data_dict_max_var)
   np.save("tmp/TabR_TabQ_trajectory", data_dict)
   return (steps, rewards)


def averageOverRuns(Agent, Env, exp):
   rewards = []
   total_steps = []
   for run in range(exp.runs):
       env = Env(exp.env_params)
       np.random.seed(run)
       random.seed(run)
       agent = Agent(env.observationShape(), env.numActions(),
                     exp.meta_parameters)
       (steps, r) = runExperiment(env, exp.env_params['episodes'], agent,
                                  False)
       rewards.append(r)
       #print("Completed a run")
       total_steps.append(steps)
       # print("Completed run %d of %d"%(, exp.runs)
   metric = np.array(total_steps)
   mean = metric.mean(axis=0)
   stderr = metric.std(axis=0) / np.sqrt(exp.runs)
   return (mean, stderr)


def confidenceInterval(mean, stderr):
   return (mean - stderr, mean + stderr)


def plotRewards(ax, rewards, stderr, label):
   (low_ci, high_ci) = confidenceInterval(rewards, stderr)
   ax.plot(rewards, label=label)
   ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)


def parse_args():
   parser = argparse.ArgumentParser("Bayesian exploration testbed")
   parser.add_argument(
       "-i", type=int, help="integer choosing parameter permutation to run")
   parser.add_argument(
       "-e", type=str, help="path to experiment description json file")
   parser.add_argument("-r", type=int, help="number of runs to complete")
   parser.add_argument(
       "-b", type=str, default='results', help="base path for saving results")
   parser.add_argument("--render", action="store_true")
   args = parser.parse_args()
   if args.b == None or args.r == None or args.i == None:
       print('Please run again using (without angle braces):')
       print('python q_learning.py -e path/to/exp.json -i <num> -r <num>')
       exit(1)
   return args


args = parse_args()
exp = ExperimentDescription(args.e, args.i, args.r)
Env = registry.getEnvironment(exp)
Agent = registry.getAgent(exp)

# In case the render command causes issues, get ride of the lines in between dash lines, and add the following:
# (rewards, stderr) = averageOverRuns(Agent, Env, exp)
# np.save("tmp/BayesianQ_mean_rewards", rewards)

# If the render command causes issues, get rid of the following lines: ---------
if args.render:
   print("Render mode")
   env = Env(exp.env_params)
   agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
   runExperiment(env, exp.env_params['episodes'], agent, args.render)
else:
   (rewards, stderr) = averageOverRuns(Agent, Env, exp)

   # np.save("tmp/BayesianQ_mean_rewards", rewards)

# save some metric for performance to file
meanResult = np.mean(rewards)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
   f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
   dump({"results": (rewards, stderr)}, f)
