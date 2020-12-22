import sys

sys.path.append('..')
sys.path.append('../..')

from util import *
from train import nonprivately_train, privately_train
from privacy_accounting import compute_eps_uniform as compute_eps
import pickle

from configs import get_config
from datasets import get_dataset
import argparse


def main(
    batch_size=64,
    dataset='mnist',
    iterations=500,
    l2_norm_clip=1.5,
    noise_multiplier=0.7,
    num_shards=250,
    num_slices=1,
    num_updates=10,
    seed=0,
    step_size=0.25,
  ):
  rng = random.PRNGKey(seed)

  # Load dataset, print shapes
  temp, rng = random.split(rng)
  X, y, X_test, y_test = get_dataset(temp, dataset)
  print('X: {}, y: {}'.format(X.shape, y.shape))
  print('X_test: {}, y_test: {}'.format(X_test.shape, y_test.shape))

  # Shard and slice dataset
  # For reference, X[0<=i<num_shards][0<=j<num_slices] refers to the i'th shard's j'th slice
  X, y = shard_and_slice(num_shards, num_slices, X, y)

  # Load model for selected dataset
  config = get_config(dataset)
  init_params, predict = config['classifier']

  def baseline_retrain(rng, X, y, X_test, y_test):
    """Define baseline retrain evaluation."""
    # Create new randomly initialized parameters
    temp, rng = random.split(rng)
    params = init_params(rng)

    # Unshard and unslice datasets
    X_full, y_full = full_dataset(X, y)

    # Nonprivately train for `iterations` on full dataset
    temp, rng = random.split(rng)
    params = nonprivately_train(temp, params, predict, X_full, y_full, iterations)

    return accuracy(params, predict, X_test, y_test)

  # Define training function (partially applies things like `batch_size`, `noise_multiplier`, etc.)
  def private_training_fn(rng, params, predict, X, y):
    return privately_train(rng, params, predict, X, y, l2_norm_clip, noise_multiplier, iterations, batch_size, step_size)

  # Perform initial training (`iterations` gradient updates for each shard)
  temp, rng = random.split(rng)
  params = get_trained_sharded_and_sliced_params(temp, init_params, predict, X, y, private_training_fn)
  print('Accuracy: {:.4f}'.format(sharded_and_sliced_accuracy(params, predict, X_test, y_test)))
  pickle.dump(params, open('private_ensemble.pkl', 'wb'))

  # TODO: Compute overall (eps, delta) for training on all partitions
  num_examples = np.concatenate(X[0]).shape[0]
  delta = 1 / (num_examples ** 1.1)
  epsilon = compute_eps(iterations, noise_multiplier, num_examples, batch_size, delta)
  print('Eps: {}, Delta: {}'.format(epsilon, delta))

  # Perform `num_updates` deletions
  for i in range(1, num_updates+1):
    print('Performing deletion {} of {}...'.format(i, num_updates))

    # Delete example and retrain individual shard
    temp, rng = random.split(rng)
    params, X, y = delete_random_index_and_retrain(temp, params, predict, X, y, private_training_fn)
    print('Accuracy: {:.4f}'.format(sharded_and_sliced_accuracy(params, predict, X_test, y_test)))

    # TODO: Update overall (eps, delta), break if we reach limit

    # Evaluate performance of retrain baseline
    temp, rng = random.split(rng)
    print('Baseline Accuracy: {:.4f}'.format(baseline_retrain(temp, X, y, X_test, y_test)))

def parse_args():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size used during optimization (default: 64).')
  parser.add_argument('--dataset', type=str, default='mnist', help='The name of the dataset.')
  parser.add_argument('--iterations', type=int, default=500, help='Number of training iterations per shard (default: 500).')
  parser.add_argument('--l2_norm_clip', type=float, default=1.5, help='L2 norm clip for DPSGD (default: 1.5).')
  parser.add_argument('--noise_multiplier', type=float, default=0.7, help='Standard deviation of Gaussian noise for DPSGD (default: 0.7).')
  parser.add_argument('--num_shards', type=int, default=250, help='Number of shards, i.e., partitions of the dataset (default: 250).')
  parser.add_argument('--num_slices', type=int, default=1, help='Number of slices, i.e., partitions of a shard (default: 1).')
  parser.add_argument('--num_updates', type=int, default=10, help='Number of random deletions (default: 10).')
  parser.add_argument('--seed', type=int, default=0, help='The seed, for reproducibility (default: 0).')
  parser.add_argument('--step_size', type=float, default=0.25, help='Learning rate during optimization (default: 0.25).')
  return vars(parser.parse_args())

if __name__ == '__main__':
  main(**parse_args())
