from utils import *
from datasets import get_dataset
from configs import get_config

import jax.numpy as np
import numpy as onp
import random as pyrandom

import argparse
from tqdm import tqdm
from collections import defaultdict


def set_reproducibility(seed):
  pyrandom.seed(seed)
  onp.random.seed(seed)
  rng = random.PRNGKey(seed)
  return rng


def main(seed=0, dataset='mnist'):
  rng = set_reproducibility(seed)

  config = get_config(dataset)
  l2_penalty = config['l2_penalty']
  num_updates = config['num_updates']
  num_rounds = config['num_rounds']
  alpha = config['alpha']
  perfect = config['perfect']
  learning_rates = config['learning_rates']

  strong = l2_penalty
  smooth = 4 - l2_penalty
  diameter = 2
  lipshitz = 1 + l2_penalty
  learning_rates += [2 / (strong + smooth)]

  X_train, y_train, X_test, y_test = get_dataset(dataset)

  # Add intercept
  if dataset != 'adult':
    X_train_ones = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((X_train_ones, X_train), 1)

    X_test_ones = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((X_test_ones, X_test), 1)

  num_train, dim = X_train.shape
  num_test = X_test.shape[0]

  epsilons = [0.25, 0.5, 0.75, 1]
  delta = 1. / (num_train ** 1.)
  delta_alpha = 2 * alpha
  sigmas = [compute_sigma(epsilon, delta, delta_alpha) for epsilon in epsilons]

  unpublished_accuracies_across_rounds = []
  published_accuracies_across_rounds = []
  retrain_accuracies_across_rounds = []

  for round_i in range(num_rounds):
    print('Starting round {}...'.format(round_i))

    temp, rng = random.split(rng)
    W_init = init_W(temp, dim)

    print('Training on full dataset...')
    W, init_iterations = train(W_init, X_train, y_train, learning_rates, l2_penalty, alpha)

    print('Initialization iterations: {}'.format(init_iterations))

    # Delete first row `num_updates` times in sequence.
    updates = [lambda X, y: delete_index(X, y) for i in range(num_updates)]

    unpublished_accuracies = []
    published_accuracies = defaultdict(list)
    retrain_accuracies = []
    max_iterations = None

    # For each update...
    for i, update in enumerate(updates):
      print('Processing update {}...'.format(i))

      # Apply update
      X_train, y_train = update(X_train, y_train)

      # Finetune on remaining points
      W, iterations = train(W, X_train, y_train, learning_rates, l2_penalty, alpha)
      unpublished_accuracy = accuracy(W, X_test, y_test)
      unpublished_accuracies.append(unpublished_accuracy)
      print('Accuracy:              {:.4f}'.format(unpublished_accuracy))

      if not max_iterations or iterations > max_iterations:
        max_iterations = iterations
      print('Max Iterations:            {}'.format(max_iterations))

      # Record performance of published model for varying epsilons
      for epsilon, sigma in zip(epsilons, sigmas):
        temp, rng = random.split(rng)
        W_published = publish(temp, W, sigma)
        published_accuracy = accuracy(W_published, X_test, y_test)
        published_accuracies[epsilon].append(published_accuracy)
        print('Accuracy (published, ε = {:.2f}):  {:.4f}'.format(epsilon, published_accuracy))

      # Record performance of retraining from initial point
      W_retrain, _ = train(W_init, X_train, y_train, learning_rates, l2_penalty, alpha, max_iterations)
      retrain_accuracy = accuracy(W_retrain, X_test, y_test)
      retrain_accuracies.append(retrain_accuracy)
      print('Accuracy (retrain):    {:.4f}'.format(retrain_accuracy))

      print('-' * 20)

    unpublished_accuracies_across_rounds.append(unpublished_accuracies)
    published_accuracies_across_rounds.append(published_accuracies)
    retrain_accuracies_across_rounds.append(retrain_accuracies)

def parse_args():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--dataset', type=str, default='gowalla', help='The name of the dataset.')
  parser.add_argument('--seed', type=int, default=0, help='The seed, for reproducibility (default: 0).')
  return vars(parser.parse_args())

if __name__ == '__main__':
  main(**parse_args())
