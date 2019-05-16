import random
import math
import argparse
import os
import numpy as np
from pickle import dump
import time
from time import sleep

from src.RLGlue.rl_glue import RlGlue
from src.utils.AgentWrapper import AgentWrapper
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
    rewards = []
    total_steps = []
    for run in range(exp.runs):
        # set random seeds before each run
        np.random.seed(run)
        random.seed(run)

        # build the environment
        env = Env(exp.env_params)

        # build the agent and wrap it with an API compatibility layer
        agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
        agent_wrapper = AgentWrapper(agent)

        # build the rl-glue instance to handle the agent-environment interface
        glue = RlGlue(agent_wrapper, env)

        (steps, r) = runExperiment(glue, exp.env_params['episodes'], False)
        rewards.append(r)
        #print("Completed a run")
        total_steps.append(steps)
        # print("Completed run %d of %d"%(, exp.runs)

    rew_array = np.array(rewards)
    total_reward_list = []
    for run in range(exp.runs):
        total_reward_list.append(rew_array[run, -1])

    mean = np.mean(total_reward_list)
    stderr = np.std(total_reward_list) / np.sqrt(exp.runs)
    print("here is the mean over all runs = ", mean)
    print("standard error = ", stderr)
    return (mean, stderr)

def parse_args():
    parser = argparse.ArgumentParser("Bayesian exploration testbed")
    parser.add_argument("-i", type=int, help="integer choosing parameter permutation to run")
    parser.add_argument("-e", type=str, help="path to experiment description json file")
    parser.add_argument("-r", type=int, help="number of runs to complete")
    parser.add_argument("-b", type=str, default='results', help="base path for saving results")
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

if args.render:
    pass
    # print("Render mode")
    # env = Env(exp.env_params)
    # agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
    # runExperiment(env, exp.env_params['episodes'], agent, args.render)
else:
    (rewards, stderr) = averageOverRuns(Agent, Env, exp)


# save some metric for performance to file
meanResult = np.mean(rewards)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (rewards, stderr)}, f)
