import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm
from jax.numpy import linalg
from jax.experimental.stax import Dense, Relu, Tanh, Conv, MaxPool, Flatten, Softmax, LogSoftmax, Sigmoid
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.nn import log_sigmoid

from mnist import mnist

from tqdm import tqdm
import itertools
import pickle

LogSigmoid = elementwise(log_sigmoid)

def model(rng):
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
    Dense(1),
    LogSigmoid,
  )
  temp, rng = random.split(rng)
  params = init_params(temp, (-1, 28, 28, 1))[1]
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

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  X, y, X_test, y_test = mnist()
  X, X_test = X.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)
  y, y_test = (np.argmax(y, 1) % 2 == 1).astype(np.float32), (np.argmax(y_test, 1) % 1 == 1).astype(np.float32)

  temp, rng = random.split(rng)
  params, predict = model(temp)

  def loss(params, batch, l2=0.05):
    X, y = batch
    y_hat = predict(params, X).reshape(-1)
    return -np.mean(y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat))

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
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
    if i % 1000 == 0:
      params = get_params(opt_state)
      print('Loss: {:.4f}'.format(loss(params, (X, y))))
  params = get_params(opt_state)
  exit()

  pickle.dump(lr_params, open('logistic_regression_params.pkl', 'wb'))
  pickle.dump(logistic_regression, open('logistic_regression.pkl', 'wb'))
  pickle.dump(fe_params, open('feature_extractor_params.pkl', 'wb'))
  pickle.dump(feature_extractor, open('feature_extractor.pkl', 'wb'))

