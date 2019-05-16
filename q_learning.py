import random
import math
import argparse
import os
import numpy as np

from src.ExperimentDescription import ExperimentDescription
import src.registry as registry
from pickle import dump
import time
from time import sleep


def runExperiment(env, num_episodes, agent, render):

  total_reward = 0
  rewards = []
  steps = []

  for episode in range(num_episodes):
    s = env.reset()
    a = agent.start(s)
    done = False
    step = 0

    while not done:
      if render:
          print("Render env")
          env.render()

      (sp, r, done, __) = env.step(a) # Note: the environment "registers" the new sp as env.pos
      agent.update(s, sp, r, a, done)
      s = sp
      a = agent.getAction(s)
      # print("State action pair", s, a)
      # print(step)
      total_reward += r
      rewards.append(total_reward) # uncomment
      step += 1

    steps.append(step)
    print("Episode", episode, " Total_Reward", total_reward)
    # agent.print()

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

    rew_array = np.array(rewards)
    total_reward_list = []
    for run in range(exp.runs):
        total_reward_list.append(rew_array[run, -1])

    # metric = np.array(total_steps[0])
    # mean = metric.mean(axis=0)
    # stderr = metric.std(axis=0) / np.sqrt(exp.runs)
    mean = np.mean(total_reward_list)
    stderr = np.std(total_reward_list) / np.sqrt(exp.runs)
    print("here is the mean over all runs = ", mean)
    print("standard error = ", stderr)
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


# save some metric for performance to file
meanResult = np.mean(rewards)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (rewards, stderr)}, f)
