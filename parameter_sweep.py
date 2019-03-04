import random
import math
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from ExperimentDescription import ExperimentDescription
import utils.registry as registry
from pickle import dump

def runExperiment(env, num_episodes, agent):
  total_reward = 0
  rewards = []
  steps = []

  for episode in range(num_episodes):
    s = env.reset()
    a = agent.start(s)
    done = False

    step = 0

    while not done:
      (sp, r, done, __) = env.step(a) # Note: the environment "registers" the new sp as env.pos
      agent.update(s, sp, r, a, done)

      s = sp # update the current state to sp
      a = agent.getAction(s) # update the current action to a
      total_reward += r
      rewards.append(total_reward)

      step += 1

    steps.append(step)

  return (steps, rewards)

def parse_args():
  parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
  parser.add_argument("-i", type=int, help="integer choosing parameter permutation to run")
  parser.add_argument("-e", type=str, help="path to experiment description json file")
  parser.add_argument("-b", type=str, default='results', help="base path for saving results")

  args = parser.parse_args()
  if args.e == None or args.i == None:
    print('Please run again using (without angle braces):')
    print('python parameter_sweep.py -e path/to/exp.json -i <num>')
    exit(1)

  return args

args = parse_args()
exp = ExperimentDescription(args.e, args.i)

Env = registry.getEnvironment(exp)
Agent = registry.getAgent(exp)

env = Env(exp.env_params)
np.random.seed(exp.run)
random.seed(exp.run)
agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
(steps, r) = runExperiment(env, exp.env_params['episodes'], agent)

# save some metric for performance to file
meanResult = np.mean(steps)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}/{exp.run}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": steps}, f)
