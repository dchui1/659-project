import numpy as np
import matplotlib.pyplot as plt


rewards = np.load('tmp/BayesianQ_mean_rewards.npy')

fig = plt.figure()
ax = plt.axes()
ax.plot(rewards, label='BayesianQ')

plt.legend()
plt.title("Number of Steps to Reach Goal")
plt.xlabel("Number of Episodes")
plt.ylabel("Number of Steps")
plt.show()
