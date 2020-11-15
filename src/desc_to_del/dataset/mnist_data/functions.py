import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm
from jax.numpy import linalg
from jax.experimental.stax import Dense, Relu, Tanh, Conv, MaxPool, Flatten, Softmax, LogSoftmax, Sigmoid
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.nn import log_sigmoid
import numpy as onp
import shutil

from mnist import mnist

from tqdm import tqdm
import itertools
import pickle

def l2_norm(tree):
  """Compute the l2 norm of a tree of arrays. Useful for weight decay."""
  leaves, _ = tree_flatten(tree)
  return np.sqrt(sum(np.vdot(x, x) for x in leaves))

def projection(tree, max_norm=1.):
  """Clip gradients stored as a tree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(tree)
  normalize = lambda g: np.where(norm < max_norm, g, g * (max_norm / norm))
  return tree_map(normalize, tree)

def accuracy(params, predict, X, y):
  """Computes the model accuracy given a dataset."""
  y_pred = (predict(params, X).reshape(-1) > 0.5).astype(np.float32)
  return np.mean(y == y_pred)

def elementwise(fun, **fun_kwargs):
  """Layer that applies a scalar function elementwise on its inputs."""
  init_fun = lambda rng, input_shape: (input_shape, ())
  apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
  return init_fun, apply_fun

LogSigmoid = elementwise(log_sigmoid)

def feature_extractor(rng):
  """Feature extraction network."""
  init_params, forward = stax.serial(
    Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    Relu,
    MaxPool((2, 2), (1, 1)),
    Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    Relu,
    MaxPool((2, 2), (1, 1)),
    Flatten,
    Dense(32),
    Relu,
    Dense(32),
  )
  temp, rng = random.split(rng)
  params = init_params(temp, (-1, 28, 28, 1))[1]
  return params, forward

def logistic_regression(rng):
  """Logistic regression."""
  init_params, forward = stax.serial(Dense(1), Sigmoid)
  temp, rng = random.split(rng)
  params = init_params(temp, (-1, 32))[1]
  return params, forward

def data_stream(rng, batch_size, X, y):
  num_complete_batches, leftover = divmod(X.shape[0], batch_size)
  num_batches = num_complete_batches + bool(leftover)
  while True:
    temp, rng = random.split(rng)
    perm = random.permutation(temp, X.shape[0])
    for i in range(num_batches):
      batch_idx = perm[i*batch_size:(i+1)*batch_size]
      yield X[batch_idx], y[batch_idx]

def compose(params, forwards, inputs):
  for param, forward in zip(params, forwards):
    inputs = forward(param, inputs)
  return inputs

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  X, y, X_test, y_test = mnist()
  X, X_test = X.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)
  y, y_test = (np.argmax(y, 1) % 2 == 1).astype(np.float32), (np.argmax(y_test, 1) % 2 == 1).astype(np.float32)

  temp, rng = random.split(rng)
  fe_params, feature_extractor = feature_extractor(temp)

  temp, rng = random.split(rng)
  lr_params, logistic_regression = logistic_regression(temp)

  params = (fe_params, lr_params)
  predict = lambda p, x: compose(p, (feature_extractor, logistic_regression), x)

  def loss(params, batch, l2=0.05):
    X, y = batch
    y_hat = predict(params, X).reshape(-1)
    return -np.mean(np.log(y * y_hat + (1. - y) * (1. - y_hat))) + (l2 / 2) * l2_norm(params[1]) ** 2.

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    params = (params[0], projection(params[1]))
    return opt_update(i, grad(loss)(params, batch), opt_state)

  iterations = 5000
  batch_size = 64
  step_size = 0.001

  opt_init, opt_update, get_params = optimizers.adam(step_size)
  opt_state = opt_init(params)

  temp, rng = random.split(rng)
  batches = data_stream(temp, batch_size, X, y)

  for i in tqdm(range(iterations)):
    opt_state = update(i, opt_state, next(batches))
  fe_params, lr_params = get_params(opt_state)

  print('LR l2 norm: {:.4f}'.format(l2_norm(lr_params)))

  print('Accuracy (train): {:.4f}'.format(accuracy((fe_params, lr_params), predict, X, y)))
  print('Accuracy (test): {:.4f}'.format(accuracy((fe_params, lr_params), predict, X_test, y_test)))

  shutil.copyfile('train.py', 'functions.py')

  # Dump model parameters
  pickle.dump(lr_params, open('lr_params.pkl', 'wb'))
  pickle.dump(fe_params, open('fe_params.pkl', 'wb'))

  # Extract features
  X_train_proj = onp.asarray(feature_extractor(fe_params, X))
  y_train_proj = onp.asarray(y)
  X_test_proj = onp.asarray(feature_extractor(fe_params, X_test))
  y_test_proj = onp.asarray(y_test)

  # Dump to files
  dim = X_train_proj.shape[1]
  onp.savetxt('X_train_d={}.csv'.format(dim), X_train_proj, fmt='%f', delimiter=',')
  onp.savetxt('y_train_d={}.csv'.format(dim), y_train_proj, fmt='%f', delimiter=',')
  onp.savetxt('X_test_d={}.csv'.format(dim), X_test_proj, fmt='%f', delimiter=',')
  onp.savetxt('y_test_d={}.csv'.format(dim), y_test_proj, fmt='%f', delimiter=',')
