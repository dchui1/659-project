import random
import argparse
import os
import numpy as np
from pickle import dump
import json

from src.RLGlue.rl_glue import RlGlue
from src.ExperimentDescription import ExperimentDescription
import src.registry as registry

# def runExperiment(glue, num_episodes, render, w):
def runExperiment(glue, num_episodes, render):
    rewards = []
    steps = []

    # obj = glue.agent.rewardApprox
    # print(obj.__dict__.keys())

    # prior_var = glue.agent.rewardApprox.scale
    # prior_var_computed = np.square(w) * max(0, glue.agent.rewardApprox.beta_0 * (glue.agent.rewardApprox.nu_0 + 1) / (
    #     glue.agent.rewardApprox.alpha_0 * glue.agent.rewardApprox.nu_0))
    # assert (prior_var == prior_var_computed)

    # t_dist_vars = []
    # bonuses = []
    # for j in range(glue.environment.numActions()):
    #     l_a = []
    #     b_a = []
    #     for i in range(glue.agent.agent.num_states):
    #         l_s = []
    #         l_s.append(prior_var)
    #         l_a.append(l_s)
    #         b_s = []
    #         b_a.append(b_s)
    #     t_dist_vars.append(l_a)
    #     bonuses.append(b_a)

    for episode in range(num_episodes):
        glue.start()
        done = False
        step = 0
        while not done:
            if render:
                print("Render env")
                glue.environment.render()

            (r, s, a, done) = glue.step()

            # s_idx = glue.agent.agent.getIndex(s)
            # t_distribution_var = glue.agent.rewardApprox.scale
            # bonus = glue.agent.rewardApprox.bonus
            # # I think it's fine not to have to use B[x] because scale and bonus just got replaces with current values for x
            # if a is not None: #have not terminated
            #     t_dist_vars[a][s_idx].append(t_distribution_var)
            #     bonuses[a][s_idx].append(bonus)

            rewards.append(glue.total_reward)
            step += 1

        steps.append(step)
        print("Episode", episode, "steps", step)
        # print("Episode", episode, "Total_Reward", glue.total_reward)

    # data_dict = {
    #         't_dist_variances': t_dist_vars,
    #         'all_bonuses': bonuses
    #     }
    # np.save("tmp/gw/bonus_variance_dict_w{}".format(w), data_dict)

    return (steps, rewards)


def averageOverRuns(Agent, Env, exp, bonus_params):
    rewards_list = []
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
        try:
            (steps, rlist) = runExperiment(glue, exp.env_params['episodes'], False, bonus_params["w"])
        except:
            (steps, rlist) = runExperiment(glue, exp.env_params['episodes'], False)

        total_steps_400eps = np.sum(steps[:400])
        ts400.append(total_steps_400eps)
        rewards_list.append(rlist) # r is a list of rewards for all the episodes in the run
        total_steps.append(steps)
        # print("Completed run %d of %d"%(, exp.runs)

    # take mean, std and stderr over runs
    mean_r = np.mean(rewards_list, axis=0)
    stddev_r = np.std(rewards_list, axis=0)
    stderr_r = stddev_r / np.sqrt(exp.runs)

    mean_s = np.mean(total_steps, axis=0)
    stddev_s = np.std(total_steps, axis=0)
    stderr_s = stddev_s / np.sqrt(exp.runs)

    mean400 = np.mean(ts400, axis=0)
    stddev400 = np.std(ts400, axis=0)
    stderr400 = stddev400 / np.sqrt(exp.runs)

    return (mean_r, stderr_r, mean_s, stderr_s, mean400, stderr400)


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
    print("w =", bonus_params["w"], "q =", bonus_params["q"])
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
    (mean_r, stderr_r, mean_s, stderr_s, mean400, stderr400) = averageOverRuns(Agent, Env, exp, bonus_params)
    try:
        np.save("tmp/gw/aver_steps_rewards_w{}_q{}_r{}_50by50".format(bonus_params["w"], bonus_params["q"], exp.runs), np.array([mean_s, stderr_s, mean_r, stderr_r]))
        np.save("tmp/gw/stats_s400_w{}_q{}_r{}_50by50".format(bonus_params["w"], bonus_params["q"], exp.runs), np.array([mean400, stderr400]))
    except:
        np.save("tmp/gw/aver_steps_rewards_egreedy_r{}_50by50".format(exp.runs), np.array([mean_s, stderr_s, mean_r, stderr_r]))
        np.save("tmp/gw/stats_s400_egreedy_r{}_50by50".format(exp.runs), np.array([mean400, stderr400]))

# save some metric for performance to file
meanResult = np.mean(mean_s)
path = f'{args.p}/{exp.name}/{exp.environment}/{exp.agent}/{exp.getParamString()}'
os.makedirs(path, exist_ok=True)
with open(f'{path}/mean.csv', 'w') as f:
    f.write(str(meanResult))

with open(f'{path}/results.pkl', 'wb') as f:
    dump({"results": (mean_s, stderr_s)}, f)
