from clean_data import clean_adult_full
from sklearn import model_selection
import matplotlib.pyplot as plt
import pickle
from random import sample
import random
import pandas as pd
import numpy as np


import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm, clip_grads
from jax.numpy import linalg
from tqdm import tqdm


def predict(W, X):
  """Forward propagation for logistic regression."""
  return nn.sigmoid(np.dot(X, W))

def loss(W, X, y, l2_penalty=0.):
  """Binary cross entropy loss with l2 regularization."""
  y_hat = predict(W, X)
  bce = y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat)
  return -np.mean(bce) + l2_penalty * l2_norm(W)

def unit_projection(W):
  """Projects model parameters to have at most l2 norm of 1."""
  return clip_grads(W, 1)

def train(W, X, y, etas=[0.5], l2_penalty=0., iters=1, alpha=0.001):
  """Simply executes several model parameter steps."""
  iteration, close = 0, False
  while not close:
    g = grad(loss)(W, X, y, l2_penalty)
    if get_distance_to_opt(g, l2_penalty) < alpha:
      close = True
    best_W, best_loss = None, None
    for eta in etas:
      W_prime = unit_projection(W - eta * g)
      loss_i = loss(W_prime, X, y, l2_penalty)
      if not best_loss or loss_i < best_loss:
        best_W, best_loss = W_prime, loss_i
    W = best_W
    iteration += 1
  return W

def process_update(W, X, y, update, train):
  """
  Updates the dataset according to some update function (e.g. append datum, delete datum) then
  finetunes the model on the resulting dataset according to some given training function.
  """
  X, y = update(X, y)
  W = train(W, X, y)
  return W, X, y

def process_updates(W, X, y, updates, train):
  """Processes a sequence of updates."""
  for update in updates:
    W, X, y = process_update(W, X, y, update, train)
  return W, X, y

def compute_sigma(num_examples, iterations, lipshitz, strong, epsilon, delta):
  """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""
  gamma = (smooth - strong) / (smooth + strong)
  numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
  denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
  return numerator / denominator

def publish(rng, W, sigma):
  """Publishing function which adds Gaussian noise with scale sigma."""
  return W + sigma * random.normal(rng, W.shape)

def accuracy(W, X, y):
  """Computes the model accuracy given a dataset."""
  y_hat = (predict(W, X) > 0.5).astype(np.int32)
  return np.mean(y_hat == y)

def delete_index(idx, *args):
  """Deletes index `idx` from each of args (assumes they all have same shape)."""
  mask = np.eye(len(args[0]))[idx] == 0.
  return (arg[mask] for arg in args)

def append_datum(data, *args):
  return (np.concatenate((arg, datum)) for arg, datum in zip(args, data))

def get_distance_to_opt(grad, l2_penalty):
  grad_norm = np.sqrt(np.sum(np.power(grad, 2)))
  alpha = (2 / l2_penalty) * grad_norm
  return alpha


if __name__ == "__main__":
  n_deletions = 5
  n_rounds = 10
  m = 0.05
  alpha = 0.001
  iters = 25
  sampling_rate = 0.1
  perfect  = False
  # eps_list = [0.25, 0.4, 0.5, 0.75, 1, 10]
  eps = 1.
  M = 0.5
  etas = [1 / (m + M), 0.01, 0.1, 0.5, 1, 5, 10]

  X, y = clean_adult_full(scale_and_center=True, normalize=True, intercept=True, sampling_rate=sampling_rate)

  """ Split train/test and create deletion sequence. """
  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
  X_train, X_test, y_train, y_test = X.values, X_test.values, y.values, y_test.values

  y_train[y_train == -1] = 0
  y_test[y_test == -1] = 0

  X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

  rng = random.PRNGKey(0)

  num_train = X.shape[0]
  num_test = X_test.shape[0]
  num_updates = 5 # Number of deletions

  init_iterations = 1000
  update_iterations = 25

  l2_penalty = 0.05
  strong = l2_penalty
  smooth = 4 - l2_penalty
  diameter = 2
  lipshitz = 1 + l2_penalty

  epsilon = 5
  delta = 1 / (num_train ** 2)

  W = np.ones((X_train.shape[1],))
  W = unit_projection(W)
  W = train(W, X_train, y_train, etas, l2_penalty, init_iterations, alpha)

  # Delete first row `num_updates` times in sequence
  updates = [lambda X, y: delete_index(0, X, y) for i in range(num_updates)]
  train_fn = lambda W, X, y: train(W, X, y, etas, l2_penalty, update_iterations, alpha)

  print('Processing updates...')
  W, _, _ = process_updates(W, X_train, y_train, updates, train_fn)
  print('Accuracy: {:.4f}\n'.format(accuracy(W, X_test, y_test)))

  sigma = compute_sigma(num_train, update_iterations, lipshitz, strong, epsilon, delta)
  print('Epsilon: {}, Delta: {}, Sigma: {:.4f}'.format(epsilon, delta, sigma))

  temp, rng = random.split(rng)
  W = publish(temp, W, sigma)
  print('Accuracy (published): {:.4f}'.format(accuracy(W, X_test, y_test)))
