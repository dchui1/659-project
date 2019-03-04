
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np

def main():
    # fileName =sys.argv[1]

    fig = plt.figure()
    ax = plt.axes()
    # plotRewards(ax, rewards, stderr, fileName)
    runResults = []
    for i in range(1, len(sys.argv)):
        fileName = sys.argv[i]
        f1 = open(fileName, 'rb')
        data_dict = pickle.load(f1)
        steps = data_dict["results"]
        runResults.append(steps)
    (steps, stderr) = averageOverRuns(runResults)
    plotRewards(ax, steps, stderr, fileName)


    plt.legend()
    plt.title("Average Number of Steps to Reach Goal across 5 Runs")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Number of Steps to Reach Goal")
    plt.show()


def plotRewards(ax, rewards, stderr, label):
  (low_ci, high_ci) = confidenceInterval(rewards, stderr)
  ax.plot(rewards, label=label)
  ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)

def confidenceInterval(mean, stderr):
  return (mean - stderr, mean + stderr)

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

  metric = np.array(runResults)
  mean = metric.mean(axis=0)
  stderr = metric.std(axis=0) / np.sqrt(len(runResults))

  return (mean, stderr)

if __name__ == "__main__":
    main()
