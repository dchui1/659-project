import random
import math
import argparse
import os
import numpy as np
from pickle import dump
import time
import json
from time import sleep
import scipy

from src.RLGlue.rl_glue import RlGlue
from src.utils.AgentWrapper import AgentWrapper
from src.utils.Bonus_Generator import BayesianBonusGenerator
from src.ExperimentDescription import ExperimentDescription
import src.registry as registry

def runExperiment(glue, num_episodes, render):
    rewards = []
    steps = []

    for episode in range(num_episodes):
        glue.start()
        done = False

        step = 0
        while not done:
            if render:
                print("Render env")
                glue.environment.render()

            (r, s, a, done) = glue.step()

            rewards.append(glue.total_reward)
            step += 1

        steps.append(step)
        # print("Episode", episode, "steps", step)
        # print("Episode", episode, "Total_Reward", glue.total_reward)

    return (steps, rewards)

def averageOverRuns(Agent, Env, exp):
    rewards_list = []
    total_steps = []
    for run in range(exp.runs):
        # set random seeds before each run
        np.random.seed(run)
        random.seed(run)

        # build the environment
        env = Env(exp.env_params)

        # build the agent and wrap it with an API compatibility layer
        agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
        agent_wrapper = BayesianBonusGenerator(agent, bonus_params)
        # build the rl-glue instance to handle the agent-environment interface
        glue = RlGlue(agent_wrapper, env)

        (steps, r) = runExperiment(glue, exp.env_params['episodes'], False)
        rewards_list.append(r[-1])
        #print("Completed a run")
        total_steps.append(steps)
        # print("Completed run %d of %d"%(, exp.runs)
    assert len(rewards_list) == exp.runs
    rew_array = np.array(rewards_list)
    print(rew_array)
    mean = np.mean(rew_array)
    std_dev = np.std(rew_array)
    stderr = scipy.stats.sem(rew_array)
    print("here is the mean over all runs = ", mean)
    print("standard dev = ", std_dev)
    print("standard err = ", stderr)
    ci = mean_confidence_interval(rew_array, stderr)
    return (mean, std_dev, stderr, ci, rew_array)


def mean_confidence_interval(data_array, se, confidence=0.95):
    n = len(data_array)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def parse_args():
    parser = argparse.ArgumentParser("Bayesian exploration testbed")
    parser.add_argument("-i", type=int, help="integer choosing parameter permutation to run")
    parser.add_argument("-a", type=str, help="path to agent description json file")
    parser.add_argument("-b", type=str, help="path to bonus description json file")
    parser.add_argument("-r", type=int, help="number of runs to complete")
    parser.add_argument("-p", type=str, default='results', help="base path for saving results")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.p == None or args.r == None or args.i == None:
        print('Please run again using (without angle braces):')
        print('python q_learning.py -e path/to/exp.json -i <num> -r <num>')
        exit(1)
    return args

args = parse_args()
bonus_path = args.b
with open(bonus_path) as f:
    bonus_params = json.load(f)

exp = ExperimentDescription(args.a, args.i, args.r)
Env = registry.getEnvironment(exp)
Agent = registry.getAgent(exp)

if args.render:
    pass
    # print("Render mode")
    # env = Env(exp.env_params)
    # agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
    # runExperiment(env, exp.env_params['episodes'], agent, args.render)
else:
    (rewards, std_dev, stderr, ci, rew_array) = averageOverRuns(Agent, Env, exp)


# save some metric for performance to file
meanResult = np.mean(rewards)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (rewards, stderr)}, f)
