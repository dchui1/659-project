
import matplotlib.pyplot as plt
import pickle
import sys

def main():
    # fileName =sys.argv[1]

    fig = plt.figure()
    ax = plt.axes()
    # plotRewards(ax, rewards, stderr, fileName)

    for i in range(1, len(sys.argv)):
        fileName = sys.argv[i]
        f1 = open(fileName, 'rb')
        data_dict = pickle.load(f1)
        (rewards, stderr) = data_dict["results"]
        print(sys.argv[i])
        plotRewards(ax, rewards, stderr, fileName)


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

if __name__ == "__main__":
    main()
