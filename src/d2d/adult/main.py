from clean_data import clean_adult_full
from sklearn import model_selection
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import pandas as pd
import numpy as onp
import random as pyrandom

import jax.numpy as np
from jax import grad, nn, random, jit
from jax.experimental import stax, optimizers
from jax.experimental.optimizers import l2_norm, clip_grads
from jax.numpy import linalg
from tqdm import tqdm
from collections import defaultdict

import brewer2mpl


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
    if get_distance_to_opt(g, l2_penalty) < alpha:
      close = True
    else:
      best_W, best_loss = None, None
      for eta in learning_rates:
        W_prime = unit_projection(W - eta * g)
        loss_i = loss(W_prime, X, y, l2_penalty)
        if not best_loss or loss_i < best_loss:
          best_W, best_loss = W_prime, loss_i
      W = best_W
    iterations += 1
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
  y_hat = 2 * (predict(W, X) > 0.5).astype(np.int32) - 1
  return np.mean(y_hat == y)

def delete_index(idx, *args):
  """Deletes index `idx` from each of args (assumes they all have same shape)."""
  mask = np.eye(len(args[0]))[idx] == 0.
  return (arg[mask] for arg in args)

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


if __name__ == "__main__":
  seed = 0

  pyrandom.seed(seed)
  onp.random.seed(seed)
  rng = random.PRNGKey(seed)

  l2_penalty = 0.05
  strong = l2_penalty
  smooth = 4 - l2_penalty
  diameter = 2
  lipshitz = 1 + l2_penalty

  num_updates = 50
  num_rounds = 50
  alpha = 0.001
  sampling_rate = 0.1
  perfect  = False
  learning_rates = [2 / (strong + smooth), 0.01, 0.1, 0.5, 1, 5, 10]


  unpublished_accuracies_across_rounds = []
  published_accuracies_across_rounds = []
  retrain_accuracies_across_rounds = []

  X, y = clean_adult_full(intercept=True, sampling_rate=sampling_rate) #scale_and_center=True, normalize=True, intercept=True, sampling_rate=sampling_rate)

  for round_i in range(num_rounds):
    print('Starting round {}...'.format(round_i))

    # Split train/test and create deletion sequence.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    num_train = X_train.shape[0]
    dim = X_train.shape[1]
    num_test = X_test.shape[0]

    epsilons = [0.25, 0.5, 0.75, 1]
    delta = 1. / (num_train ** 1.)
    delta_alpha = 2 * alpha
    sigmas = [compute_sigma(epsilon, delta, delta_alpha) for epsilon in epsilons]
    # print('Epsilon: {}, Delta: {}, Sigma: {:.4f}'.format(epsilon, delta, sigma))

    temp, rng = random.split(rng)
    W_init = init_W(temp, dim)

    print('Training on full dataset...')
    W, _ = train(W_init, X_train, y_train, learning_rates, l2_penalty, alpha)

    # Delete first row `num_updates` times in sequence.
    updates = [lambda X, y: delete_index(0, X, y) for i in range(num_updates)]

    unpublished_accuracies = []
    published_accuracies = defaultdict(list)
    retrain_accuracies = []

    # For each update...
    for i, update in enumerate(updates):
      print('Processing update {}...'.format(i))

      # Apply update
      X_train, y_train = update(X_train, y_train)

      # Finetune on remaining points
      W, iterations = train(W, X_train, y_train, learning_rates, l2_penalty, alpha)
      unpublished_accuracy = accuracy(W, X_test, y_test)
      unpublished_accuracies.append(unpublished_accuracy)

      # Record performance of published model for varying epsilons
      for epsilon, sigma in zip(epsilons, sigmas):
        temp, rng = random.split(rng)
        W_published = publish(temp, W, sigma)
        published_accuracy = accuracy(W_published, X_test, y_test)
        published_accuracies[epsilon].append(published_accuracy)

      # Record performance of retraining from initial point
      W_retrain, _ = train(W_init, X_train, y_train, learning_rates, l2_penalty, alpha, iterations)
      retrain_accuracy = accuracy(W_retrain, X_test, y_test)
      retrain_accuracies.append(retrain_accuracy)

      print('Iterations:            {}'.format(iterations))
      print('Accuracy:              {:.4f}'.format(unpublished_accuracy))
      print('Accuracy (published):  {:.4f}'.format(published_accuracy))
      print('Accuracy (retrain):    {:.4f}'.format(retrain_accuracy))
      print('-' * 20)

    unpublished_accuracies_across_rounds.append(unpublished_accuracies)
    published_accuracies_across_rounds.append(published_accuracies)
    retrain_accuracies_across_rounds.append(retrain_accuracies)

  unpublished_accuracies = [sum([unpublished_accuracies_across_rounds[i][j] for i in range(num_rounds)]) / num_rounds for j in range(num_updates)]
  published_accuracies = defaultdict(list)
  for epsilon in epsilons:
    published_accuracies[epsilon] = [sum([published_accuracies_across_rounds[round][epsilon][update] for round in range(num_rounds)]) / num_rounds for update in range(num_updates)]
  retrain_accuracies = [sum([retrain_accuracies_across_rounds[i][j] for i in range(num_rounds)]) / num_rounds for j in range(num_updates)]

  pickle.dump(unpublished_accuracies, open('unpublished_accuracies.pkl', 'wb'))
  pickle.dump(published_accuracies, open('published_accuracies.pkl', 'wb'))
  pickle.dump(retrain_accuracies, open('retrain_accuracies.pkl', 'wb'))

  """
  unpublished_accuracies =  pickle.load(open('unpublished_accuracies.pkl', 'rb'))
  published_accuracies = pickle.load(open('published_accuracies.pkl', 'rb'))
  retrain_accuracies =  pickle.load(open('retrain_accuracies.pkl', 'rb'))
  """

  plt.rc('font', family='sans-serif')
  plt.rc('xtick', labelsize='x-small')
  plt.rc('ytick', labelsize='x-small')

  colors = iter(brewer2mpl.get_map('Paired', 'qualitative', 6).mpl_colors)
  linestyles = iter([
    'solid',
    (0, (1, 1)), # densely dotted
    (0, (5, 1)), # densely dashed
    (0, (3, 1, 1, 1)), # densely dashdotted
    (0, (3, 1, 1, 1, 1, 1)), # densely dashdotdotted
    (0, (1, 1)), # dotted
    (0, (3, 5, 1, 5)), # dashdotted
    (0, (3, 5, 1, 5, 1, 5)), # dashdotdotted
    (0, (5, 5)), # dashed
    (0, (1, 10)), #loosely dotted
    (0, (5, 10)), #loosely dashed
    (0, (3, 10, 1, 10)), #loosely dashdotted
    (0, (3, 10, 1, 10, 1, 10)), # loosely dashdotdotted
  ])

  plt.rc('font', family='sans-serif')
  plt.rc('xtick', labelsize='x-small')
  plt.rc('ytick', labelsize='x-small')

  fig = plt.figure(figsize=(4, 3))
  ax = fig.add_subplot()

  ax.plot(
    list(range(num_updates)),
    unpublished_accuracies,
    label='Unpublished',
    color=next(colors),
    linestyle=next(linestyles),
  )

  for epsilon in reversed(sorted(epsilons)):
    ax.plot(
      list(range(num_updates)),
      published_accuracies[epsilon],
      label=r'Published ($\varepsilon$ = {})'.format(epsilon),
      color=next(colors),
      linestyle=next(linestyles),
    )

  ax.plot(
    list(range(num_updates)),
    retrain_accuracies,
    label='Retrain',
    color=next(colors),
    linestyle=next(linestyles),
  )

  ax.set_xlabel(r'Number of Deletions ($\delta = {:.2e}$)'.format(delta))
  ax.set_ylabel(r'Test Accuracy')

  ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  ax.grid(True)
  ax.legend(loc='lower right', prop={'size': 6})
  ax.set_rasterized(True)

  fig.tight_layout()
  plt.savefig('plot.png', dpi=400)
