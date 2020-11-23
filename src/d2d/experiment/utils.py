import pandas as pd
import numpy as onp
import random as pyrandom

import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm, clip_grads
from jax.numpy import linalg

def predict(W, X):
  """Forward propagation for logistic regression."""
  return nn.sigmoid(np.dot(X, W))

def loss(W, X, y, l2_penalty=0.):
  """Log loss."""
  log_loss = np.mean(np.log(1 + np.exp(-y * np.dot(X, W))))
  penalty = 0.5 * l2_penalty * np.sum(np.power(W, 2))
  return log_loss + penalty

def unit_projection(W):
  """Projects model parameters to have at most l2 norm of 1."""
  return W / np.max((1., np.sqrt(np.sum(np.power(W, 2)))))

def gradient_loss_fn(W, X, y, l2_penalty):
  n = X.shape[0]
  log_grad = np.dot(np.diag(-y/(1 + np.exp(y*np.dot(X, W)))), X)
  log_grad_sum = np.dot(np.ones(n), log_grad)
  reg_grad = l2_penalty * W
  return (reg_grad + (1/n)*log_grad_sum)

@jit
def gradient(W, X, y, l2_penalty):
  return grad(loss)(W, X, y, l2_penalty)

def train(W, X, y, learning_rates=[0.5], l2_penalty=0., alpha=0.001, max_iterations=None):
  """Simply executes several model parameter steps."""
  iterations, close = 0, False
  while not close:
    if max_iterations and iterations == max_iterations:
      return W, iterations
    g = gradient(W, X, y, l2_penalty)
    dist_to_opt = get_distance_to_opt(g, l2_penalty)
    print(dist_to_opt)
    print(accuracy(W, X, y))
    print('-' * 20)
    if dist_to_opt < alpha:
      close = True
    else:
      best_W, best_loss, best_eta = None, None, None
      for eta in learning_rates:
        W_prime = unit_projection(W - eta * g)
        loss_i = loss(W_prime, X, y, l2_penalty)
        if not best_loss or loss_i < best_loss:
          best_W, best_loss, best_eta = W_prime, loss_i, eta
      W = best_W
    iterations += 1
  exit()
  return W, iterations

def process_update(W, X, y, update, train):
  """
  Updates the dataset according to some update function (e.g. append datum, delete datum) then
  finetunes the model on the resulting dataset according to some given training function.
  """
  X, y = update(X, y)
  W, iterations = train(W, X, y)
  return W, X, y, iterations

def process_updates(W, X, y, updates, train):
  """Processes a sequence of updates."""
  for update in tqdm(updates):
    W, X, y, _ = process_update(W, X, y, update, train)
  return W, X, y

def compute_sigma(epsilon, delta_0, delta_1):
  """
  def compute_sigma(num_examples, iterations, lipshitz, strong, epsilon, delta):
    # Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf
    gamma = (smooth - strong) / (smooth + strong)
    numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
    denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
    return numerator / denominator
  """
  numerator = delta_1 / np.sqrt(2)
  denominator = np.sqrt(np.log(1. / delta_0) + epsilon) - np.sqrt(np.log(1. / delta_0))
  return numerator / denominator

def publish(rng, W, sigma):
  """Publishing function which adds Gaussian noise with scale sigma."""
  noise = sigma * random.normal(rng, W.shape)
  W = W + noise
  W = unit_projection(W)
  return W

def accuracy(W, X, y):
  """Computes the model accuracy given a dataset."""
  y_hat = 2 * ((predict(W, X) > 0.5) - 0.5)
  return np.mean(y_hat == y)

def delete_index(*args):
  """Deletes index `idx` from each of args (assumes they all have same shape)."""
  return (arg[1:] for arg in args)

def append_datum(data, *args):
  return (np.concatenate((arg, datum)) for arg, datum in zip(args, data))

def get_distance_to_opt(grad, l2_penalty):
  grad_norm = np.sqrt(np.sum(np.power(grad, 2)))
  alpha = (2. / l2_penalty) * grad_norm
  return alpha

def init_W(rng, dim):
  """
  W = np.ones((dim,))
  W = unit_projection(W)
  """
  temp, rng = random.split(rng)
  W = random.normal(temp, (dim,))
  W = unit_projection(W)
  temp, rng = random.split(rng)
  W = random.uniform(temp, ()) * W
  return W

