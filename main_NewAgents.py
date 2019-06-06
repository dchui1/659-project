import random
import math
import argparse
import os
import numpy as np
from pickle import dump
import time
import json
from time import sleep

from src.RLGlue.rl_glue import RlGlue
from src.ExperimentDescription import ExperimentDescription
import src.registry as registry


def make_nested_list(num_states=6):
    nested_list = []
    list = []
    for i in range(num_states):
        nested_list.append(list)
    return nested_list


def runExperiment(glue, num_episodes, render, bonus_params, run):
    rewards = []
    steps = []
    num_states = glue.environment.observationShape()[0]
    state_visits_all_episodes = np.zeros(num_states)
    for episode in range(num_episodes):
        glue.start()
        done = False
        step = 0
        while not done:
            if render:
                print("Render env")
                glue.environment.render()
            (r, s, a, done) = glue.step()
            state_idx = s[0]
            state_visits_all_episodes[state_idx] += 1
            rewards.append(glue.total_reward)
            step += 1
        steps.append(step)
    state_visits_all_episodes = np.array(state_visits_all_episodes/np.sum(state_visits_all_episodes)) * 100
    return (steps, rewards, state_visits_all_episodes)


def averageOverRuns(Agent, Env, exp, bonus_params):
    rewards = []
    total_steps = []
    sv_allruns = []
    for run in range(exp.runs):
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
        (steps, r, sv_run) = runExperiment(glue, exp.env_params['episodes'], False, bonus_params, run=run)

        rewards.append(r)
        total_steps.append(steps)
        sv_allruns.append(sv_run)
        print("completed run ", run)

    sv_allruns = np.array(sv_allruns)
    # np.save("tmp/rs/s_visits_allruns_q{}".format(bonus_params["q"]), sv_allruns)

    # rew_array = np.array(rewards)
    # total_reward_list = []
    # for run in range(exp.runs):
    #     total_reward_list.append(rew_array[run, -1])
    #
    # mean_rew = np.mean(total_reward_list)
    # stderr_rew = np.std(total_reward_list) / np.sqrt(exp.runs)
    # print("here is the mean over all runs = ", mean_rew)
    # print("standard error = ", stderr_rew)
    return total_steps # fix this!


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
if bonus_path is not None:
    with open(bonus_path) as f:
        bonus_params = json.load(f)
    print("quantile = ", bonus_params["q"])
else:
    bonus_params = {"name": "default"}

exp = ExperimentDescription(args.a, args.i, args.r)
Env = registry.getEnvironment(exp)
Agent = registry.getAgent(exp)


if args.render:
    pass
else:
    (rewards, stderr_rew) = averageOverRuns(Agent, Env, exp, bonus_params)
    # np.save("tmp/rs/aver_epis_q{}_w{}".format(bonus_params["q"], bonus_params["w"]), np.array([rewards, stderr_rew]))


# save some metric for performance to file
print(type(rewards))
meanResult = np.mean(rewards)
path = f'{args.p}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (rewards, stderr_rew)}, f)
