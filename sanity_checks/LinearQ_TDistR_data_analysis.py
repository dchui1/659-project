import numpy as np

# "normal_prior_log_precision": [-0.01, 0.05, 0.1, 0.15]
data_dict = np.load('tmp/debugging.npy')
X = data_dict.item().get('X')
y = data_dict.item().get('y')

XtX = np.dot(np.transpose(X), X)


prior_scalar_precision = np.exp(-0.01)
prior_precision = np.eye(X.shape[1]) * prior_scalar_precision
a = np.linalg.inv(XtX + prior_precision)

mu_0 = np.repeat(0.0, X.shape[1])
mu_0 = mu_0.reshape(len(mu_0), 1)
b = (np.dot(prior_precision, mu_0) + np.dot(np.transpose(X), y))

mu_n = np.dot(a, b)

print(mu_n.shape)

# compute R:
epsilon = np.random.normal(0.0, 1.0, (X.shape[0], 1))
R = np.dot(X, mu_n) + epsilon

print(R.shape)
