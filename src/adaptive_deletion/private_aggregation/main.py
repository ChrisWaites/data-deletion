import sys

sys.path.append('..')
sys.path.append('../..')

from sharding import *
import models
from train import train
import pickle
import matplotlib.pyplot as plt
import brewer2mpl

from configs import get_config
from datasets import get_dataset
import argparse

def main(
    dataset='mnist',
    seed=0,
    num_shards=250,
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

  init_params, predict = config['clf']

  try:
    params = pickle.load(open('private_aggregation.pkl', 'rb'))
  except:
    print('Training full model (Shards={}, Slices={})...'.format(num_shards, num_slices))
    # params[0 <= i < num_shards][0 <= j <= num_slices] refers to the params trained on the first j slices of the i'th shard,
    # i.e., j == 0 yields randomly initialized params trained on no data, j == 1 yields params trained on the first slice, etc.
    params = get_trained_sharded_and_sliced_params(rng, init_params, predict, X, y, train)
    pickle.dump(params, open('private_aggregation.pkl', 'wb'))

  targets = np.argmax(y_test, axis=1)
  predictions = sharded_and_sliced_predict(params, predict, X_test)
  nonprivate_accuracy = np.mean(predictions == targets)
  print('Accuracy (nonprivate): {:.4}\n'.format(nonprivate_accuracy))

  print('Example votes:')
  print(get_votes(params, predict, X_test)[:20])

  mechanism_names = ['Exp. Mech.', 'LNMax']
  mechanisms = [exponential_mechanism, lnmax]
  mechanism_accs = []
  per_example_epsilons = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]

  for mechanism in mechanisms:
    epsilon_accs = []
    print('Mechanism: {}'.format(mechanism))
    for per_example_epsilon in per_example_epsilons:
      temp, rng = random.split(rng)
      agg = lambda votes: mechanism(rng, votes, per_example_epsilon)
      predictions = sharded_and_sliced_predict(params, predict, X_test, agg)
      accuracy = np.mean(predictions == targets)
      print('Accuracy (eps={:.4}): {:.4}\n'.format(per_example_epsilon, accuracy))
      epsilon_accs.append(accuracy)
    mechanism_accs.append(epsilon_accs)

def parse_args():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--dataset', type=str, default='mnist', help='The name of the dataset.')
  parser.add_argument('--seed', type=int, default=0, help='The seed, for reproducibility (default: 0).')
  return vars(parser.parse_args())

if __name__ == '__main__':
  main(**parse_args())

