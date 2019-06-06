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
            # b = glue.agent.rewardApprox.b
            # print("bonus = ", b)
            step += 1
        tr = glue.total_reward
        rewards.append(tr)
        steps.append(step)
        print("Episode", episode, "steps", step, "total reward", tr)
        # print("Episode", episode, "Total_Reward", glue.total_reward)

    return (steps, rewards) # steps is a list of #steps required to complete each episode
    # i.e., steps[i] = # steps to complete episode i, for every i


def averageOverRuns(Agent, Env, exp):
    rewards_list_allruns = []
    total_steps = []
    ts400 = []
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
        (steps, rlist) = runExperiment(glue, exp.env_params['episodes'], False)

        total_steps_400eps = np.sum(steps[:400])
        ts400.append(total_steps_400eps)
        rewards_list_allruns.append(rlist) # r is a list of rewards for all the episodes in the run
        total_steps.append(steps)
        print("Completed run ", run)

    rew_array_allruns = np.array(rewards_list_allruns)
    step_array_allruns = np.array(total_steps)
    means_s = step_array_allruns.mean(axis=0)
    stderrs_s = step_array_allruns.std(axis=0) / np.sqrt(exp.runs)
    means_r = rew_array_allruns.mean(axis=0)
    stderrs_r = rew_array_allruns.std(axis=0) / np.sqrt(exp.runs)

    return (means_s, stderrs_s, means_r, stderrs_r, ts400)


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
    # print("quantile =", bonus_params["q"], " w =", bonus_params["w"])
else:
    bonus_params = {"name": "default"}

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
    (mean_s, stderr_s, mean_r, stderr_r, ts400) = averageOverRuns(Agent, Env, exp)
    # np.save("tmp/rs/aver_epis_q{}_w{}_r{}".format(bonus_params["q"], bonus_params["w"], exp.runs), np.array([mean_s, stderr_s, mean_r, stderr_r]))

m_ts400 = np.mean(ts400)
stderr_ts400 = np.std(ts400)/np.sqrt(exp.runs)
np.save("tmp/gw/stats_s400_w{}_q{}_r{}".format(bonus_params["w"], bonus_params["q"], exp.runs), np.array([m_ts400, stderr_ts400]))


# save some metric for performance to file
meanResult_s = np.mean(mean_s)
path = f'{args.p}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult_s))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (mean_s, stderr_s)}, f)
