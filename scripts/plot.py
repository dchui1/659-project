
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np

def main():
    # fileName =sys.argv[1]

    fig = plt.figure()
    ax = plt.axes()
    # plt.ylim(0.0, 50000) # only use for GridWorld experiments (keep the same ylim)
    # plotRewards(ax, rewards, stderr, fileName)
    runResults = []
    for i in range(1, len(sys.argv)):
        fileName = sys.argv[i]
        f1 = open(fileName, 'rb')
        data_dict = pickle.load(f1)
        (steps, stderr) = data_dict["results"]
        # runResults.append(steps)
    # (steps, stderr) = averageOverRuns(runResults)
        plotRewards(ax, steps, stderr, fileName.replace(".pkl", ""))


    plt.legend()
    plt.title("Average Number of Steps to Reach Goal")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Number of Steps to Reach Goal")
    plt.show()


def plotRewards(ax, rewards, stderr, label):
  (low_ci, high_ci) = confidenceInterval(rewards, stderr)
  ax.plot(rewards, label=label)
  ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)

def confidenceInterval(mean, stderr):
  return (mean - stderr, mean + stderr)



if __name__ == "__main__":
    main()
