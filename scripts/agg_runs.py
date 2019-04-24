import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    args = parse_args()
    path = args.path


    (mean, stderr) = aggregate_results(path)
    if args.save:
        print("Save called")
        agent = args.agent
        with open(f'{agent}.pkl', 'wb') as f:
            pickle.dump({"results": (mean, stderr)}, f)

    fig = plt.figure()
    ax = plt.axes()
    plotRewards(ax, mean, stderr, "title")


    plt.legend()
    plt.title("Average Number of Steps to Reach Goal across 5 Runs")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Number of Steps to Reach Goal")
    plt.show()


def aggregate_results(path):

    dirs = os.listdir(path)
    print("Dirs length", len(dirs))
    all_steps = []

    for dir in dirs:
        fileName = path + dir + "/results.pkl"
        f = open(fileName, 'rb')
        data_dict = pickle.load(f)
        steps = data_dict["results"]

        all_steps.append(steps)

    (mean, stderr) = averageOverRuns(all_steps)

    return (mean, stderr)

    # print(mean, stderr)
    # print(all_steps)

def averageOverRuns(runResults):

  # rewards = []
  # total_steps = []
  # for run in runResults:
  #   env = Env(exp.env_params)
  #   np.random.seed(run)
  #   random.seed(run)
  #   agent = Agent(env.observationShape(), env.numActions(), exp.meta_parameters)
  #   (steps, r) = runExperiment(env, exp.env_params['episodes'], agent)
  #   rewards.append(r)
  #   print("Completed a run")
  #   total_steps.append(steps)
    # print("Completed run %d of %d"%(, exp.runs)
  # print(len(runResults), "run results length")

  # print(len(runResults[0]))
  # print(len(runResults[50]))

  metric = np.array(runResults)
  # print("Metric shape", metric.shape)
  mean = metric.mean(axis=0)

  stderr = metric.std(axis=0) / np.sqrt(len(runResults))

  return (mean, stderr)




def plotRewards(ax, rewards, stderr, label):
    (low_ci, high_ci) = confidenceInterval(rewards, stderr)
    ax.plot(rewards, label=label)
    ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)

def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def parse_args():
    parser = argparse.ArgumentParser("Process results")
    parser.add_argument("-path", type=str, help="path to parameter sweep")
    parser.add_argument("-agent", type=str, help="name of the agent")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    main()
