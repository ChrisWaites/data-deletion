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
  return bce + l2 * l2_norm(params)

def unit_projection(params):
  """Projects model parameters to have at most l2 norm of 1."""
  return clip_grads(params, 1)

def step(params, predict, X, y, l2=0., proj=unit_projection):
  """A single step of projected gradient descent."""
  opt_init, opt_update, get_params = optimizers.sgd(step_size=0.5)
  opt_state = opt_init(params)
  opt_state = opt_update(0, grad(loss)(params, predict, X, y, l2), opt_state)
  params = get_params(opt_state)
  params = proj(params)
  return params

def train(params, predict, X, y, l2=0., iters=1):
  """Simply executes several model parameter steps."""
  for i in range(iters):
    params = step(params, predict, X, y, l2)
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
  for update in updates:
    params, X, y = process_update(params, X, y, update, train)
  return params, X, y

def compute_sigma(num_examples, iterations, lipshitz, strong, epsilon, delta):
  """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""
  gamma = (smooth - strong) / (smooth + strong)
  numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
  denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
  return numerator / denominator

def publish(rng, params, sigma):
  """Publishing function which adds Gaussian noise with scale sigma."""
  flat_params, tree_def = tree_flatten(params)
  rngs = random.split(rng, len(flat_params))
  noised_params = [param + sigma * random.normal(rng, param.shape) for rng, param in zip(rngs, flat_params)]
  return tree_unflatten(tree_def, noised_params)

def accuracy(params, X, y):
  """Computes the model accuracy given a dataset."""
  return np.mean((predict(params, X) > 0.5).astype(np.int32) == y)

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

  num_examples = X_test.shape[0]
  num_updates = 25

  init_iterations = 1000
  update_iterations = 25

  feature_extractor_init, feature_extractor = stax.serial(
    Conv(16, (8, 8), padding='SAME', strides=(2, 2)), Relu, MaxPool((2, 2), (1, 1)),
    Conv(32, (4, 4), padding='VALID', strides=(2, 2)), Relu, MaxPool((2, 2), (1, 1)),
    Flatten, Dense(32), Relu,
  )

  temp, rng = random.split(rng)
  feature_extractor_params = feature_extractor_init(temp, (-1, 28, 28, 1))[1]

  classifier_init, classifier = stax.serial(
    Dense(10), LogSoftmax,
  )

  temp, rng = random.split(rng)
  classifier_params = classifier_init(temp, (-1, 32))[1]

  def pretrain(params, predict, X, y, iterations=75, batch_size=32, step_size=0.15):
    def data_stream():
      num_train = X.shape[0]
      num_complete_batches, leftover = divmod(num_train, batch_size)
      num_batches = num_complete_batches + bool(leftover)

      while True:
        perm = random.permutation(temp, num_train)
        for i in range(num_batches):
          batch_idx = perm[i*batch_size:(i+1)*batch_size]
          yield X[batch_idx], y[batch_idx]

    def loss(params, batch):
      inputs, targets = batch
      preds = predict(params, inputs)
      return -np.mean(np.sum(preds * targets, axis=1))

    @jit
    def update(i, opt_state, batch):
      params = get_params(opt_state)
      return opt_update(i, grad(loss)(params, batch), opt_state)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)

    batches = data_stream()
    itercount = itertools.count()
    for _ in range(iterations):
      opt_state = update(next(itercount), opt_state, next(batches))
    return get_params(opt_state)

  def full_predict(params, inputs):
    feature_extractor_params, classifer_params = params
    features = feature_extractor(feature_extractor_params, inputs)
    return classifier(classifer_params, features)

  feature_extractor_params, params = pretrain((feature_extractor_params, classifier_params), full_predict, X, y)

  l2 = 0.05

  strong = l2
  smooth = 4 - l2
  diameter = 2
  lipshitz = 1 + l2

  epsilon = 5
  delta = 1 / (num_examples ** 2)

  # Throw out training data, fix feature extractor
  X, y = X_test, y_test

  def predict(params, X):
    features = feature_extractor(feature_extractor_params, X)
    return classifier(params, features)

  # Delete first row `num_updates` times in sequence
  updates = [lambda X, y: delete_index(0, X, y) for i in range(num_updates)]
  train_fn = lambda params, X, y: train(params, predict, X, y, l2, update_iterations)

  params, X, y = process_updates(params, X, y, updates, train_fn)
  print('Before publishing: {:.4f}'.format(accuracy(params, X, y)))

  sigma = compute_sigma(num_examples, update_iterations, lipshitz, strong, epsilon, delta)
  temp, rng = random.split(rng)
  params = publish(temp, params, sigma)

  print('Epsilon: {}, Delta: {}, Sigma: {:.4f}'.format(epsilon, delta, sigma))
  print('After publishing: {:.4f}'.format(accuracy(params, X, y)))
