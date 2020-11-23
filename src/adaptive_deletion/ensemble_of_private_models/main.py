import sys

sys.path.append('..')
sys.path.append('../..')

from sharding import *
import models
from train import privately_train
from privacy_accounting import compute_eps_uniform as compute_eps
import pickle

from configs import get_config
from datasets import get_dataset
import argparse

def main(
    dataset='mnist',
    seed=0,
    num_shards=20,
    num_slices=1,
  ):

  rng = random.PRNGKey(seed)

  X, y, X_test, y_test = get_dataset(dataset)
  config = get_config(dataset)

  temp, rng = random.split(rng)
  X, y = shuffle(temp, X, y)

  print('X: {}, y: {}'.format(X.shape, y.shape))
  print('X_test: {}, y_test: {}'.format(X_test.shape, y_test.shape))

  # X[0<=i<num_shards][0<=j<num_slices] refers to the j'th slice of the i'th shard
  X, y = shard_and_slice(num_shards, num_slices, X, y)

  l2_norm_clip = config['l2_norm_clip']
  noise_multiplier = config['noise_multiplier']
  iterations = config['iterations']
  batch_size = config['batch_size']
  step_size = config['step_size']

  train = lambda rng, params, predict, X, y: privately_train(rng, params, predict, X, y, l2_norm_clip, noise_multiplier, iterations, batch_size, step_size)

  N = np.concatenate(X[0]).shape[0]
  delta = 1 / (N ** 1.1)
  epsilon = compute_eps(iterations, noise_multiplier, N, batch_size, delta)
  print(epsilon, delta)

  init_params, predict = config['clf']

  try:
    params = pickle.load(open('private_ensemble.pkl', 'rb'))
  except:
    print('Training full model (Shards={}, Slices={})...'.format(num_shards, num_slices))
    # params[0 <= i < num_shards][0 <= j <= num_slices] refers to the params trained on the first j slices of the i'th shard,
    # i.e., j == 0 yields randomly initialized params trained on no data, j == 1 yields params trained on the first slice, etc.
    params = get_trained_sharded_and_sliced_params(rng, init_params, predict, X, y, train)
    pickle.dump(params, open('private_ensemble.pkl', 'wb'))

  targets = np.argmax(y_test, axis=1)
  predictions = sharded_and_sliced_predict(params, predict, X_test)
  print('Accuracy (ε = {:.4f}, δ = {}): {:.4}\n'.format(epsilon, delta, np.mean(predictions == targets)))

  print(get_votes(params, predict, X_test)[:20])

def parse_args():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--dataset', type=str, default='mnist', help='The name of the dataset.')
  parser.add_argument('--seed', type=int, default=0, help='The seed, for reproducibility (default: 0).')
  return vars(parser.parse_args())

if __name__ == '__main__':
  main(**parse_args())
