import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm
from jax.numpy import linalg
from jax.experimental.stax import Dense, Relu, Tanh, Conv, MaxPool, Flatten, Softmax, LogSoftmax
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from mnist import mnist

from tqdm import tqdm
import itertools
import pickle


def loss(params, predict, X, y, l2=0.):
  """Binary cross entropy loss with l2 regularization."""
  y_hat = predict(params, X)
  return -np.mean(np.sum(y * y_hat, axis=1)) + (l2 / 2) * l2_norm(params) ** 2.

def l2_norm(tree):
  """Compute the l2 norm of a tree of arrays. Useful for weight decay."""
  leaves, _ = tree_flatten(tree)
  return np.sqrt(sum(np.vdot(x, x) for x in leaves))

def projection(tree, max_norm=1.):
  """Clip gradients stored as a tree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(tree)
  normalize = lambda g: np.where(norm < max_norm, g, g * (max_norm / norm))
  return tree_map(normalize, tree)

def train(params, step, iters=1):
  """Executes several model parameter steps."""
  for i in range(iters):
    params = step(params)
  return params

def process_update(params, X, y, update, train):
  """
  Updates the dataset according to some update function (e.g. append datum, delete datum) then
  finetunes the model on the resulting dataset according to some given training function.
  """
  X, y = update(X, y)
  params = train(params, X, y)
  return params, X, y

def process_updates(params, X, y, updates, train):
  """Processes a sequence of updates."""
  for update in tqdm(updates):
    params, X, y = process_update(params, X, y, update, train)
  return params, X, y

def compute_sigma(num_examples, iterations, lipshitz, strong, diameter, epsilon, delta):
  """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""
  gamma = (smooth - strong) / (smooth + strong)
  numerator = 4 * np.sqrt(2) * (lipshitz + smooth * diameter) * np.power(gamma, iterations)
  denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
  return numerator / denominator

def publish(rng, params, sigma):
  """Publishing function which adds Gaussian noise with scale sigma."""
  params, tree_def = tree_flatten(params)
  rngs = random.split(rng, len(params))
  noised_params = [param + sigma * random.normal(rng, param.shape) for rng, param in zip(rngs, params)]
  return tree_unflatten(tree_def, noised_params)

def accuracy(params, predict, X, y):
  """Computes the model accuracy given a dataset."""
  y_pred = np.argmax(predict(params, X), 1)
  y = np.argmax(y, 1)
  return np.mean(y == y_pred)

def delete_index(idx, *args):
  """Deletes index `idx` from each of args (assumes they all have same shape)."""
  mask = np.eye(len(args[0]))[idx] == 0.
  return (arg[mask] for arg in args)

def append_datum(data, *args):
  return (np.concatenate((arg, datum)) for arg, datum in zip(args, data))

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
  )
  temp, rng = random.split(rng)
  params = init_params(temp, (-1, 28, 28, 1))[1]
  return params, forward

def logistic_regression(rng):
  """Logistic regression."""
  init_params, forward = stax.serial(Dense(10), LogSoftmax)
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

def pretrain(rng, params, predict, loss, X, y, iterations=5000, batch_size=64, step_size=0.001):
  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    params = (params[0], projection(params[1]))
    return opt_update(i, grad(loss)(params, batch), opt_state)

  opt_init, opt_update, get_params = optimizers.adam(step_size)
  opt_state = opt_init(params)

  temp, rng = random.split(rng)
  batches = data_stream(temp, batch_size, X, y)
  for i in tqdm(range(iterations)):
    opt_state = update(i, opt_state, next(batches))

  return get_params(opt_state)

def compose(params, forwards, inputs):
  for param, forward in zip(params, forwards):
    inputs = forward(param, inputs)
  return inputs

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  X, y, X_test, y_test = mnist()
  X, X_test = X.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)

  temp, rng = random.split(rng)
  fe_params, feature_extractor = feature_extractor(temp)

  temp, rng = random.split(rng)
  lr_params, logistic_regression = logistic_regression(temp)

  params = (fe_params, lr_params)
  predict = lambda p, x: compose(p, [feature_extractor, logistic_regression], x)

  def pretrain_loss(params, batch, l2=0.05):
    _, lr_params = params
    X, y = batch
    y_hat = predict(params, X)
    return -np.mean(np.sum(y * y_hat, axis=1)) + (l2 / 2) * l2_norm(lr_params) ** 2.

  try:
    print('Loading model...')
    lr_params = pickle.load(open('lr_params.pkl', 'rb'))
    fe_params = pickle.load(open('fe_params.pkl', 'rb'))
  except:
    print('Training model on public points...')
    temp, rng = random.split(rng)
    fe_params, lr_params = pretrain(temp, params, predict, pretrain_loss, X, y)
    pickle.dump(lr_params, open('lr_params.pkl', 'wb'))
    pickle.dump(fe_params, open('fe_params.pkl', 'wb'))

  def predict(lr_params, X):
    features = feature_extractor(fe_params, X)
    return logistic_regression(lr_params, features)

  print('\nAccuracy on public points: {:.4f}'.format(accuracy(lr_params, predict, X, y)))
  print('Accuracy on private points: {:.4f}'.format(accuracy(lr_params, predict, X_test, y_test)))

  # Throw out public points, only keep feature extractor
  X, y = X_test, y_test

  print('Number of private points: {}'.format(X.shape[0]))

  epsilon = 1.
  delta = 1 / (X.shape[0] ** 2.)

  print('\nEpsilon: {}'.format(epsilon))
  print('Delta: {}'.format(delta))

  projection_radius = 1.
  l2 = 0.05
  strong = l2
  smooth = 4 - l2
  diameter = 2 * projection_radius
  lipshitz = 1 + l2
  gamma = (smooth - strong) / (smooth + strong)

  update_iterations = 150
  init_iterations = int(update_iterations + np.log((diameter * strong * X.shape[0]) / (2 * lipshitz)) / np.log(1. / gamma))

  print('\nInitialization iterations: {}'.format(init_iterations))
  print('Update iterations: {}'.format(update_iterations))
  print('L2: {}'.format(l2))

  @jit
  def step(params):
    """A single step of projected gradient descent."""
    gs = tree_flatten(grad(loss)(params, predict, X, y, l2))[0]
    params, tree_def = tree_flatten(params)
    params = [param - (2 / (strong + smooth)) * g for param, g in zip(params, gs)]
    params = projection(params)
    return tree_unflatten(tree_def, params)

  print('\nFinetuning on private points...')
  for i in range(init_iterations):
    lr_params = step(lr_params)
  print('Accuracy (not published): {:.4f}'.format(accuracy(lr_params, predict, X, y)))

  sigma = compute_sigma(X.shape[0], update_iterations, lipshitz, strong, diameter, epsilon, delta)
  print('Sigma: {:.4f}'.format(sigma))
  temp, rng = random.split(rng)
  print('Accuracy (published): {:.4f}'.format(accuracy(publish(temp, lr_params, sigma), predict, X, y)))

  num_updates = 25
  # Delete first row `num_updates` times in sequence
  updates = [lambda X, y: delete_index(0, X, y) for i in range(num_updates)]
  train_fn = lambda params, X, y: train(params, step, update_iterations)

  print('\nProcessing deletion requests...')
  lr_params, X, y = process_updates(lr_params, X, y, updates, train_fn)
  print('Accuracy (not published): {:.4f}'.format(accuracy(lr_params, predict, X, y)))

  sigma = compute_sigma(X.shape[0], update_iterations, lipshitz, strong, diameter, epsilon, delta)
  print('Sigma: {:.4f}'.format(sigma))
  temp, rng = random.split(rng)
  print('Accuracy (published): {:.4f}'.format(accuracy(publish(temp, lr_params, sigma), predict, X, y)))
