import numpy as np
import matplotlib.pyplot as plt

# "normal_prior_log_precision": [-0.01, 0.05, 0.1, 0.15]
data_dict = np.load('tmp/debugging.npy')
X = data_dict.item().get('X')
y = data_dict.item().get('y')
ig = data_dict.item().get('ig_scale')

fig = plt.figure()
ax = plt.axes()

plt.plot(ig)

plt.title("IG_scale for each time-step")
plt.xlabel("Number of Time-Steps across Episodes")
plt.ylabel("IG_scale values")

plt.show()


# XtX = np.dot(np.transpose(X), X)
# XtX = (0.005 * np.eye(X.shape[1])) + XtX
# XtX_inv = np.linalg.inv(XtX)
# XtX_inv_t = np.transpose(XtX_inv)
# Xt = np.transpose(X)
# right = np.dot(XtX_inv_t, Xt)
#
#
# Z = np.dot(X, right)
# np.save('tmp/debugging_Z.npy')
