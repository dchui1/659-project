'''
The goal of this script is to take the s, a, r visited by the TabularQ agent,
and compute the posterior-distribution parameters of the LinearQ_TdistR agent.
'''

import numpy as np
import tensorflow as tf
import argparse
import os
from utils.ExperimentDescription import ExperimentDescription
import utils.registry as registry


data_dict = np.load("tmp/debugging_Xy_TabularAgent")
X = data_dict.item().get('X')
y = data_dict.item().get('y')
# plug it into the bayesian approximator for the LinearQ_TdistR agent


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
env = Env(exp.env_params)
Agent = registry.getAgent(exp)


def agent_factory():
    return Agent(env.observationShape(), env.numActions(), exp.meta_parameters)


new_agent = agent_factory()
new_agent.rewardApprox.update_stats(X, y)
normal_covariance = new_agent.rewardApprox.T_distribution.mnig_prior.normal_prior.covariance_scale.numpy()
normal_precision = new_agent.rewardApprox.T_distribution.mnig_prior.normal_prior.precision.numpy()
ig_scale = new_agent.rewardApprox.T_distribution.mnig_prior.ig_prior.scale.numpy()
ig_shape = new_agent.rewardApprox.T_distribution.mnig_prior.ig_prior.shape.numpy()
