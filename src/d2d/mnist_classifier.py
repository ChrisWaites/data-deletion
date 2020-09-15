import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm, clip_grads
from jax.numpy import linalg
from jax.experimental.stax import Dense, Relu, Tanh, Conv, MaxPool, Flatten, Softmax, LogSoftmax
from jax.tree_util import tree_flatten, tree_unflatten

from mnist import mnist

from tqdm import tqdm
import itertools


def loss(params, predict, X, y, l2=0.):
  """Binary cross entropy loss with l2 regularization."""
  y_hat = predict(params, X)
  bce = -np.mean(np.sum(y * y_hat, axis=1))
  return bce + (l2 / 2) * l2_norm(params)

def projection(params, norm=1.):
  """Projects model parameters to have at most l2 norm of 1."""
  return clip_grads(params, norm)

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

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  X, y, X_test, y_test = mnist()
  X, X_test = X.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)

  fe_init, fe_forward = stax.serial(
    Conv(16, (8, 8), padding='SAME', strides=(2, 2)), Relu, MaxPool((2, 2), (1, 1)),
    Conv(32, (4, 4), padding='VALID', strides=(2, 2)), Relu, MaxPool((2, 2), (1, 1)),
    Flatten, Dense(32), Relu,
  )
  temp, rng = random.split(rng)
  fe_params = fe_init(temp, (-1, 28, 28, 1))[1]

  lr_init, lr_forward = stax.serial(Dense(10), LogSoftmax)
  temp, rng = random.split(rng)
  lr_params = lr_init(temp, (-1, 32))[1]

  def pretrain(params, predict, X, y, iterations=5000, batch_size=64, step_size=0.001):
    def data_stream():
      rng = random.PRNGKey(0)

      num_complete_batches, leftover = divmod(X.shape[0], batch_size)
      num_batches = num_complete_batches + bool(leftover)
      while True:
        temp, rng = random.split(rng)
        perm = random.permutation(temp, X.shape[0])
        for i in range(num_batches):
          batch_idx = perm[i*batch_size:(i+1)*batch_size]
          yield X[batch_idx], y[batch_idx]

    def loss(params, batch):
      X, y = batch
      y_hat = predict(params, X)
      return -np.mean(np.sum(y * y_hat, axis=1))

    @jit
    def update(i, opt_state, batch):
      params = get_params(opt_state)
      return opt_update(i, grad(loss)(params, batch), opt_state)

    def accuracy(params, batch):
      X, y = batch
      y = np.argmax(y, 1)
      y_hat = np.argmax(predict(params, X), 1)
      return np.mean(y_hat == y)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)

    batches = data_stream()
    for i in tqdm(range(iterations)):
      opt_state = update(i, opt_state, next(batches))
    return get_params(opt_state)

  def full_predict(params, inputs):
    fe_params, lr_params = params
    features = fe_forward(fe_params, inputs)
    return lr_forward(lr_params, features)

  print('Training feature extractor...\n')
  fe_params, params = pretrain((fe_params, lr_params), full_predict, X, y)

  def predict(params, X):
    features = fe_forward(fe_params, X)
    return lr_forward(params, features)

  print('Accuracy (public): {:.4f}'.format(accuracy(params, predict, X, y)))
  print('Accuracy (private): {:.4f}\n'.format(accuracy(params, predict, X_test, y_test)))

  # Throw out public points
  X, y = X_test, y_test

  print('Number of private points: {}'.format(X.shape[0]))

  projection_radius = 4

  l2 = 0.05
  strong = l2
  smooth = 4 - l2
  diameter = 2 * projection_radius
  lipshitz = 1 + l2

  epsilon = 10.
  delta = 1 / (X.shape[0] ** 1.1)

  init_iterations = 100
  update_iterations = 75
  num_updates = 25

  @jit
  def step(params):
    """A single step of projected gradient descent."""
    gs = tree_flatten(grad(loss)(params, predict, X, y, l2))[0]
    params, tree_def = tree_flatten(params)
    params = [param - (2 / (strong + smooth)) * g for param, g in zip(params, gs)]
    params = projection(params, projection_radius)
    return tree_unflatten(tree_def, params)

  print('Training on private points...')
  params = train(params, step, init_iterations)
  print('Accuracy (not published): {:.4f}'.format(accuracy(params, predict, X, y)))

  sigma = compute_sigma(X.shape[0], update_iterations, lipshitz, strong, diameter, epsilon, delta)
  print('Epsilon: {}, Delta: {}, Sigma: {:.4f}'.format(epsilon, delta, sigma))
  temp, rng = random.split(rng)
  published_params = publish(temp, params, sigma)
  print('Accuracy (published): {:.4f}\n'.format(accuracy(published_params, predict, X, y)))

  # Delete first row `num_updates` times in sequence
  updates = [lambda X, y: delete_index(0, X, y) for i in range(num_updates)]
  train_fn = lambda params, X, y: train(params, step, update_iterations)

  print('Processing updates...')
  params, X, y = process_updates(params, X, y, updates, train_fn)
  print('Accuracy (not published): {:.4f}'.format(accuracy(params, predict, X, y)))

  sigma = compute_sigma(X.shape[0], update_iterations, lipshitz, strong, diameter, epsilon, delta)
  print('Epsilon: {}, Delta: {}, Sigma: {:.4f}'.format(epsilon, delta, sigma))
  temp, rng = random.split(rng)
  published_params = publish(temp, params, sigma)
  print('Accuracy (published): {:.4f}\n'.format(accuracy(published_params, predict, X, y)))
