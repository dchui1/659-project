import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

w_vec = [1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0]
means = []
errs = []
runs = 5
# 1e8 results computed for 40 runs
for w in w_vec:
    metrics = np.load('tmp/rs/aver_epis_q0.99_w{}_r{}.npy'.format(w, runs))
    print(metrics)
    print(metrics.shape)
    mean = metrics[2]
    stderr = metrics[3]
    means.append(mean)
    errs.append(stderr)

print(means)
print(errs)
fig = plt.figure()
ax = plt.axes()
plt.plot(w_vec, means)
# plt.yscale("log")
plt.title("Average Return across Runs for RiverSwim")
plt.xlabel("w values")
plt.ylabel("Average Total Reward")
plt.grid(True)
# plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.show()
