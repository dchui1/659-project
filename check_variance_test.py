import numpy as np
import matplotlib.pyplot as plt

# "normal_prior_log_precision": [-0.01, 0.05, 0.1, 0.15]
# Z = np.load('tmp/debugging_Z.npy')
var_dict = np.load('tmp/debugging_var1.npy')
ts = var_dict.item().get('ts')
var = var_dict.item().get('var')
igsc = var_dict.item().get('igsc')

fig = plt.figure()
ax = plt.axes()

plt.plot(ts, igsc)

plt.title("IG_scale for each time-step")
plt.xlabel("Number of Time-Steps across Episodes")
plt.ylabel("IG_scale values")

plt.show()
