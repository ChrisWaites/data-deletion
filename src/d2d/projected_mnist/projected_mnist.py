import jax.numpy as np
from numpy import genfromtxt

def projected_mnist():
  X_train = genfromtxt('X_train.csv', delimiter=',')
  y_train = 2 * (genfromtxt('y_train.csv', delimiter=',') - 0.5)

  X_test = genfromtxt('X_test.csv', delimiter=',')
  y_test = 2 * (genfromtxt('y_test.csv', delimiter=',') - 0.5)

  return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

