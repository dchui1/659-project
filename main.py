import random
import argparse
import os
import numpy as np
from pickle import dump
import json

from src.RLGlue.rl_glue import RlGlue
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
        print("Episode", episode, "steps", step)
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
        AgentWrapper = registry.getAgentWrapper(bonus_params["name"])
        agent_wrapper = AgentWrapper(agent, bonus_params)
        # build the rl-glue instance to handle the agent-environment interface
        glue = RlGlue(agent_wrapper, env)

        (steps, r) = runExperiment(glue, exp.env_params['episodes'], False)
        rewards_list.append(r)
        #print("Completed a run")
        total_steps.append(steps)
        # print("Completed run %d of %d"%(, exp.runs)

    # take mean / std over runs
    mean = np.mean(rewards_list, axis=0)
    std_dev = np.std(rewards_list, axis=0, ddof=1)
    stderr = std_dev / np.sqrt(exp.runs)

    print("here is the mean over all runs = ", mean)
    print("standard dev = ", std_dev)
    print("standard err = ", stderr)

    return (mean, stderr)


def parse_args():
    parser = argparse.ArgumentParser("Bayesian exploration testbed")
    parser.add_argument("-i", type=int, help="integer choosing parameter permutation to run")
    parser.add_argument("-a", type=str, help="path to agent description json file")
    parser.add_argument("-b", type=str, help="path to bonus description json file")
    parser.add_argument("-r", type=int, help="number of runs to complete")
    parser.add_argument("-p", type=str, default='results', help="base path for saving results")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.a == None or args.r == None or args.i == None:
        print('Please run again using (without angle braces):')
        print('python q_learning.py -a path/to/agent.json -b path/to/bonus.json -i <num> -r <num>')
        exit(1)
    return args

args = parse_args()
bonus_path = args.b
if bonus_path is not None:
    with open(bonus_path) as f:
        bonus_params = json.load(f)
else:
    bonus_params = {"name": "default"}

exp = ExperimentDescription(args.a, args.i, args.r)
Env = registry.getEnvironment(exp)
Agent = registry.getAgent(exp)

if args.render:
    # pass
    print("Render mode")
    env = Env(exp.env_params)
    agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
    AgentWrapper = registry.getAgentWrapper(bonus_params["name"])
    agent_wrapper = AgentWrapper(agent, bonus_params)
    # build the rl-glue instance to handle the agent-environment interface
    glue = RlGlue(agent_wrapper, env)
    runExperiment(glue, exp.env_params['episodes'], args.render)
else:
    (mean, stderr) = averageOverRuns(Agent, Env, exp)


# save some metric for performance to file
meanResult = np.mean(mean)
path = f'{args.p}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (mean, stderr)}, f)
