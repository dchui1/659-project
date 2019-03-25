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
print("here")
from utils.bayesianapproximator import *
from utils.BNNApproximation import BNNApproximation
from utils.ExperimentDescription import ExperimentDescription
import utils.registry as registry
from pickle import dump
from agents.TabularQ import TabularQ


def runExperiment(env, num_episodes, agent, render):
    print("inside_run_exp")
    total_reward = 0
    rewards = []
    steps = []
    num_states = np.prod(env.observationShape())
    num_acts = env.numActions()
    ac_vals_list = []

    agent.bayesianQ.mu_0  # prior sample mean
    agent.bayesianQ.nu_0  # prior "observations that make the prior mean"
    agent.bayesianQ.alpha_0  # prior IG shape
    agent.bayesianQ.beta_0  # prior IG scale
    # variance according to the wiki page:
    prior_var = agent.bayesianQ.beta_0 * (agent.bayesianQ.nu_0 + 1) / (
        agent.bayesianQ.alpha_0 * agent.bayesianQ.nu_0)
    t_dist_vars = []
    t_dist_var_s0a2 = []
    for j in range(num_acts):
        l_a = []
        for i in range(num_states):
            l_s = []
            l_s.append(prior_var)
            l_a.append(l_s)
        t_dist_vars.append(l_a)

    # show the variance upto time T (ask Dustin/Mike)
    max_var_T = []
    max_m2m1 = []
    max_m1mu = []
    for episode in range(num_episodes):
        s = env.reset()
        a = agent.start(s)
        done = False
        step = 0

        while not done:
            if render:
                print("Render env")
                env.render()

            (sp, r, done, __) = env.step(a)  # Note: the environment "registers" the new sp as env.pos
            agent.update(s, sp, r, a, done)
            # compute the variance of the t-distribution (as in the formula in wiki) for the state sp. This is the state in which the bayesian distribution is updated.
            sp_idx = np.ravel_multi_index(sp, agent.state_shape)
            # x = agent.getIndex(sp) + (a * num_states)
            # nu = agent.bayesianQ.B[x, 1]
            # alpha = agent.bayesianQ.B[x, 2]
            # beta = agent.bayesianQ.B[x, 3]
            # t_distribution_var = beta * (nu + 1) / (alpha * nu)
            # t_dist_vars[a][sp_idx].append(t_distribution_var)
            ac_vals_list.append(agent.act_vals)

            m2_minus_m1sqrd_timestep = []
            var_single_timestep = []
            m1_minus_mu_timestep = []
            for x_i in range(num_states * num_acts):
                mu_i = agent.bayesianQ.B[x_i, 0]
                nu_i = agent.bayesianQ.B[x_i, 1]
                alpha_i = agent.bayesianQ.B[x_i, 2]
                beta_i = agent.bayesianQ.B[x_i, 3]
                t_distribution_var = beta_i / (nu_i * (alpha_i - 1))
                # print(mu_i)
                # print(nu_i)
                # print(alpha_i)
                # print(beta_i)
                print(t_distribution_var)
                print(agent.bayesianQ.gamma)
                print(done)
                print("")
                var_single_timestep.append(t_distribution_var)
                m2_minus_m1sqrd_timestep.append(agent.bayesianQ.m2_minus_m1_sqrd)
            max_v = np.max(var_single_timestep)
            max_var_T.append(max_v)
            max_m2m1.append(np.max(m2_minus_m1sqrd_timestep))

            if sp_idx == 0 and a == 2:
                t_dist_var_s0a2.append(t_distribution_var)

            data_dict = {
                    'max_var_T_array': max_var_T,
                    't_dist_variances_s0a2': t_dist_var_s0a2,
                    'm2m1': max_m2m1
                }

            s = sp  # update the current state to sp
            a = agent.getAction(s)  # update the current action to a
            # print(agent.act_vals)
            total_reward += r
            rewards.append(total_reward)
            step += 1

        steps.append(step)
        # print("Episode", episode, " Step", step)
        # agent.print()
    np.save("tmp/Ac_Vals", ac_vals_list)
    np.save("tmp/BayesianQ_t_dist_variances", data_dict)
    print("run_experiment ran fine")
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
        print("Completed a run")
        total_steps.append(steps)
        # print("Completed run %d of %d"%(, exp.runs)
    metric = np.array(total_steps)
    mean = metric.mean(axis=0)
    stderr = metric.std(axis=0) / np.sqrt(exp.runs)
    print("averageOverRunas ran fine")
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
(rewards, stderr) = averageOverRuns(Agent, Env, exp)
np.save("tmp/BayesianQ_mean_rewards", rewards)
# fig = plt.figure()
# ax = plt.axes()
# plotRewards(ax, rewards, stderr, "BayesianQ-Agent")

# save some metric for performance to file
meanResult = np.mean(rewards)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (rewards, stderr)}, f)

# plt.legend()
# plt.title("Average Number of Steps to Reach Goal across 1 Runs")
# plt.xlabel("Number of Episodes")
# plt.ylabel("Average Number of Steps to Reach Goal")
# plt.show()

# if args.render:
#     print("Render mode")
#     env = Env(exp.env_params)
#     agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
#     runExperiment(env, exp.env_params['episodes'], agent, args.render)
# else:
#     (rewards, stderr) = averageOverRuns(Agent, Env, exp)
#
#     fig = plt.figure()
#     ax = plt.axes()
#     plotRewards(ax, rewards, stderr, 'Bayesian-Q')
#
# # save some metric for performance to file
#     meanResult = np.mean(rewards)
#     path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
#     os.makedirs(path, exist_ok=True)
#     with open(f'{path}/mean.csv', 'w') as f:
#         f.write(str(meanResult))
#
#     with open(f'{path}/results.pkl', 'wb') as f:
#         dump({"results": (rewards, stderr)}, f)
#
#     plt.legend()
#     plt.title("Average Number of Steps to Reach Goal across 1 Run")
#     plt.xlabel("Number of Episodes")
#     plt.ylabel("Average Number of Steps to Reach Goal")
#     plt.show()
