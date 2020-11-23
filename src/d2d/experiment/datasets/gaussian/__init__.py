import jax.numpy as np
from jax import random

from sklearn import datasets, preprocessing

def gaussian():
  n_samples = 30000
  n_train = int(0.9 * n_samples)

  X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=2, random_state=0)

  y = 2 * (y - 0.5)

  X_train, X_test = X[:n_train], X[n_train:]
  y_train, y_test = y[:n_train], y[n_train:]

  scaler = preprocessing.StandardScaler()

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

