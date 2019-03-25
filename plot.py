import numpy as np
import matplotlib.pyplot as plt
import datetime

rewards = np.load('tmp/BayesianQ_mean_rewards.npy')
print(rewards.shape)
# def confidenceInterval(mean, stderr):
#   return (mean - stderr, mean + stderr)
#
# def plotRewards(ax, rewards, stderr, label):
#   (low_ci, high_ci) = confidenceInterval(rewards, stderr)
#   ax.plot(rewards, label=label)
#   ax.fill_between(range(rewards.shape[0]), low_ci, high_ci, alpha=0.4)

fig = plt.figure()
ax = plt.axes()
ax.plot(rewards, label='BayesianQ')
# plotRewards(ax, rewards, stderr, "BayesianQ-Agent")

plt.legend()
plt.title("Average Number of Steps to Reach Goal across 1 Runs")
plt.xlabel("Number of Episodes")
plt.ylabel("Average Number of Steps to Reach Goal")

plt.savefig('./tmp/' + str(datetime.datetime.now().time()) + '.png')
#plt.show()

