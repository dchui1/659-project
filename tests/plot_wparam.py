import numpy as np
import matplotlib.pyplot as plt


w_vec = [1.0, 5000.0, 100000.0, 500000.0, 1000000.0, 1.0e8]
means = []
errs = []
# 1e8 results computed for 40 runs
for w in w_vec:
    w=w_vec[2]
    metrics = np.load('tmp/rs/aver_epis_q0.99_w{}.npy'.format(w))
    print(metrics)
    if metrics.shape == ():
        print("only single value")
        means.append(float(metrics))
    else:
        print("multiple values")
        mean = metrics[0]
        # stderr = metrics[1]
        means.append(mean)
        # errs.append(stderr)

print(means)
fig = plt.figure()
ax = plt.axes()
ax.plot(w_vec, means)
plt.title("Average Return across Runs")
plt.xlabel("w values")
plt.ylabel("Average Total Reward")
plt.show()
