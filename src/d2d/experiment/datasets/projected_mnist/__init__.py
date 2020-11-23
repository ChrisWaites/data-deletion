from numpy import genfromtxt
import jax.numpy as np

def projected_mnist():
  X_train = genfromtxt('datasets/projected_mnist/X_train.csv', delimiter=',')
  y_train = 2 * (genfromtxt('datasets/projected_mnist/y_train.csv', delimiter=',') - 0.5)

  X_test = genfromtxt('datasets/projected_mnist/X_test.csv', delimiter=',')
  y_test = 2 * (genfromtxt('datasets/projected_mnist/y_test.csv', delimiter=',') - 0.5)

  return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
