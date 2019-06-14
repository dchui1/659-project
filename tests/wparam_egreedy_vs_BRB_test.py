# The following script compared the total number of steps taken by BRB and egreedy to complete 400 episodes, wrt 'w' param.
# environment = gw
# Input: array = [mean, stdderr] for each w parameter
# Output: plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale


w_vec = [0.01, 0.05, 0.07, 0.075, 0.08, 0.1, 0.5, 1.0, 2.0, 5.0]
# w_vec_r1 =
means_ws = []
errs_ws = []

for w in w_vec:
    try:
        runs = 5
        metrics = np.load('tmp/gw/stats_s400_w{}_q0.99_r{}.npy'.format(w, runs))
    except:
        runs = 1
        metrics = np.load('tmp/gw/stats_s400_w{}_q0.99_r{}.npy'.format(w, runs))
    # print(metrics)
    # print(metrics.shape)
    means_ws.append(metrics[0])
    errs_ws.append(metrics[1])

print(means_ws)
print(errs_ws)

# retrieve results for e-greedy
runs = 1
metrics_egreedy = np.load('tmp/gw/stats_s400_egreedy_r{}.npy'.format(runs))
means_egreedy = metrics_egreedy[0]

fig = plt.figure()
ax = plt.axes()
# yerr = np.array(errs_ws)
plt.plot(w_vec, means_ws, '-o', label='BRB_q0.99_epsilon0.05')
plt.axhline(y=means_egreedy, color='r', linestyle='-', label='egreedy_epsilon0.05')
plt.xlim(0.0, 1.0)
# plt.errorbar(w_vec, means_ws, yerr=yerr, label='BRB_q0.99')
# plt.errorbar(w_vec, y + 2, yerr=yerr, label='egreedy')

plt.title("Number of Steps to Complete 400 Episodes across Runs")
plt.xlabel("w values")
plt.ylabel("Number of Steps")
plt.grid(True)
plt.legend()
plt.show()
