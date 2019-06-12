import numpy as np
import matplotlib.pyplot as plt


data_dict = np.load('tmp/BayesianQ_t_dist_variances.npy')
max_var_T = data_dict.item().get('max_var_T_array')
print(min(max_var_T))

fig = plt.figure()
ax = plt.axes()
ax.plot(max_var_T)


plt.title("Maximum T-Distribution Variances Across All Episodes")
plt.xlabel("Timesteps")
plt.ylabel("Maximumum Variance over all (s, a) pairs")
plt.show()
