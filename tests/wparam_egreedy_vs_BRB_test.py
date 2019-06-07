# The following script compared the total number of steps taken by BRB and egreedy to complete 400 episodes, wrt 'w' param.
# environment = gw
# Input: array = [mean, stdderr] for each w parameter
# Output: plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale


w_vec = w_vec = [1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0]
runs = 5
means_ws = []
errs_ws = []

for w in w_vec:
    metrics = np.load('tmp/gw/stats_s400_w{}_q0.99_r{}.npy'.format(w, runs))
    print(metrics)
    print(metrics.shape)
    means_ws.append(metrics[0])
    errs_ws.append(metrics[1])

print(means_ws)
print(errs_ws)
fig = plt.figure()
ax = plt.axes()
yerr = np.array(errs_ws)
plt.plot(w_vec, means_ws)
# plt.errorbar(w_vec, means_ws, yerr=yerr, label='BRB_q0.99')

# plt.errorbar(w_vec, y + 2, yerr=yerr, label='egreedy')

# plt.yscale("log")
plt.title("Average Number of Steps to Complete 400 Episodes across Runs")
plt.xlabel("w values")
plt.ylabel("Average Number of Steps")
plt.grid(True)
# plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.show()
