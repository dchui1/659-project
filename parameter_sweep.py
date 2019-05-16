import random
import math
import argparse
import os
import numpy as np
from pickle import dump

from src.RLGlue.rl_glue import RlGlue
from src.utils.AgentWrapper import AgentWrapper
from src.ExperimentDescription import ExperimentDescription
import src.registry as registry

def runExperiment(glue, num_episodes):
    rewards = []
    steps = []

    for episode in range(num_episodes):
        glue.start()
        done = False

        step = 0
        while not done:
            (r, s, a, done) = glue.step()

            rewards.append(glue.total_reward)
            step += 1

        steps.append(step)
        # print("Episode", episode, "steps", step)
        # print("Episode", episode, "Total_Reward", glue.total_reward)

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

# set random seeds before each run
np.random.seed(exp.run)
random.seed(exp.run)

# build the environment
env = Env(exp.env_params)

# build the agent and wrap it with an API compatibility layer
agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
agent_wrapper = AgentWrapper(agent)

# build the rl-glue instance to handle the agent-environment interface
glue = RlGlue(agent_wrapper, env)

(steps, r) = runExperiment(glue, exp.env_params['episodes'])

# save some metric for performance to file
meanResult = np.mean(steps)
path = f'{args.b}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}/{exp.run}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": steps}, f)
