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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils.bayesianapproximator import *
from utils.BNNApproximation import BNNApproximation
from utils.ExperimentDescription import ExperimentDescription
from utils.bayesianapproximator import TDistBayesianApproximation
import utils.registry as registry
from pickle import dump
tf.enable_eager_execution()


def check_algebra(X):
    size = X.shape
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtX_inv = np.linalg.inv(XtX)
    Z = X @ np.transpose(XtX_inv) @ Xt
    # Id = np.eye(size[0])
    min_el = np.min(Z)
    max_el = np.max(Z)
    return (min_el, max_el)


def is_diagonal(matrix):
    #create a dummy matrix
    dummy_matrix = np.ones(matrix.shape)
    # Fill the diagonal of dummy matrix with 0.
    np.fill_diagonal(dummy_matrix, 0)
    return np.count_nonzero(np.multiply(dummy_matrix, matrix)) == 0


def runExperiment(env, num_episodes, agent, agent_factory):
    total_reward = 0
    rewards = []
    steps = []
    variances_BLR = []
    scales_BLR = []
    total_steps = [0]

    for episode in range(num_episodes):
        s = env.reset()
        a = agent.start(s)
        done = False
        step = 0

        while not done:
            (sp, r, done, __) = env.step(a)
            total_steps.append(total_steps[-1] + 1)
            agent.update(s, sp, r, a, done)
            # try:
            #     agent.update(s, sp, r, a, done)
            #     # normal_covariance = agent.rewardApprox.T_distribution.mnig_prior.normal_prior.covariance_scale.numpy()
            #     # ig_scale = agent.rewardApprox.T_distribution.mnig_prior.ig_prior.scale.numpy()
            #     # scales_BLR.append(ig_scale)
            #     #
            #     # assert(is_diagonal(normal_covariance))
            #     # max_var = max(np.diag(normal_covariance))
            #     # variances_BLR.append(max_var)
            #
            # except ArithmeticError:
            #     print("Arithmetic exception raised.")
            #     print("Episode", episode, " Step", step)
            #     data_mat = np.vstack(agent.data)
            #     rewards_mat = np.vstack(agent.reward_data)
            #
            #     new_agent = agent_factory()
            #     new_agent.rewardApprox.update_stats(data_mat, rewards_mat)
            #     normal_covariance = new_agent.rewardApprox.T_distribution.mnig_prior.normal_prior.covariance_scale.numpy()
            #     normal_precision = new_agent.rewardApprox.T_distribution.mnig_prior.normal_prior.precision.numpy()
            #     ig_scale = new_agent.rewardApprox.T_distribution.mnig_prior.ig_prior.scale.numpy()
            #     ig_shape = new_agent.rewardApprox.T_distribution.mnig_prior.ig_prior.shape.numpy()
            #
            #     scales_BLR.append(ig_scale)
            #     assert (is_diagonal(normal_covariance))
            #     max_var = max(np.diag(normal_covariance))
            #     variances_BLR.append(max_var)
            #
            #     data_dict = {
            #         'normal_covariance': normal_covariance,
            #         'normal_precision': normal_precision,
            #         'ig_scale': ig_scale,
            #         'ig_shape': ig_shape,
            #         'X': data_mat,
            #         'y': rewards_mat
            #     }
            #
            #     np.save("tmp/debugging", data_dict)
            #     exit()

            s = sp
            # ac_vals = agent.action_values(s)
            # print(ac_vals)
            a = agent.getAction(s)
            total_reward += r
            rewards.append(total_reward)
            step += 1

            # for a in range(env.numActions()):
            #     plt.imshow(agent.rewardApprox.action_var[a].reshape((30, 30)), cmap='hot', vmin=0.0, vmax=0.1)
            #     plt.savefig(f'figs/heat_map.{ss)tep}.{a}.png')

        steps.append(step)

        print("Episode", episode, " Step", step)

    # data_mat = np.vstack(agent.data)
    # rewards_mat = np.vstack(agent.reward_data)
    # data_dict = {'X': data_mat, 'y': rewards_mat}
    # np.save("tmp/debugging_Xy_TabularAgent", data_dict)

    return (steps, rewards, total_steps[1:], variances_BLR, scales_BLR)


def averageOverRuns(Agent, Env, exp):
    rewards = []
    total_steps = []
    for run in range(exp.runs):
        env = Env(exp.env_params)
        np.random.seed(run)
        tf.random.set_random_seed(run)
        random.seed(run)
        agent = Agent(env.observationShape(), env.numActions(),
                      exp.meta_parameters)

        def agent_factory():
            return Agent(env.observationShape(), env.numActions(),
                         exp.meta_parameters)

        (steps, r, ts, variances, ig_scale) = runExperiment(
            env, exp.env_params['episodes'], agent, agent_factory)
        rewards.append(r)
        print("Completed a run")
        total_steps.append(steps)
    min_step = min(steps)
    print("min_step = ")
    print(min_step)
    metric = np.array(total_steps)
    mean = metric.mean(axis=0)
    stderr = metric.std(axis=0) / np.sqrt(exp.runs)

    return (mean, stderr, ts, variances, ig_scale)


def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)


def plotRewards(ax, rewards, stderr, label):
    (low_ci, high_ci) = confidenceInterval(rewards, stderr)
    ax.plot(rewards, label=label)
    ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)


def parse_args():
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments")
    parser.add_argument(
        "-i", type=int, help="integer choosing parameter permutation to run")
    parser.add_argument(
        "-e", type=str, help="path to experiment description json file")
    parser.add_argument("-r", type=int, help="number of runs to complete")
    parser.add_argument(
        "-b", type=str, default='results', help="base path for saving results")

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

(rewards, stderr, ts, var, ig_scale) = averageOverRuns(Agent, Env, exp)

fig = plt.figure()
ax = plt.axes()
plotRewards(ax, rewards, stderr, "TabularRTabularQ")

# var_dict = {'ts': ts, 'var': var, 'igsc': ig_scale}
# np.save("tmp/debugging_var1", var_dict)

# save some metric for performance to file
meanResult = np.mean(rewards)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (rewards, stderr)}, f)

plt.legend()
plt.title("Average Number of Steps to Reach Goal across 1 Runs")
plt.xlabel("Number of Episodes")
plt.ylabel("Average Number of Steps to Reach Goal")
plt.show()
