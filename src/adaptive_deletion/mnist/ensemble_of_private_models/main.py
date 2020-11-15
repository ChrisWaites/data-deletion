import sys

sys.path.append('..')
sys.path.append('../..')

from sharding import *
import models
from train import privately_train, private_training_parameters
from privacy_accounting import compute_eps_uniform
import pickle

if __name__ == '__main__':
  rng = random.PRNGKey(0)
  num_shards, num_slices = 20, 1

  X, y, X_test, y_test = mnist()
  X, X_test = X.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)

  temp, rng = random.split(rng)
  X, y = shuffle(temp, X, y)

  print('X: {}, y: {}'.format(X.shape, y.shape))
  print('X_test: {}, y_test: {}'.format(X_test.shape, y_test.shape))

  # X[0<=i<num_shards][0<=j<num_slices] refers to the j'th slice of the i'th shard
  X, y = shard_and_slice(num_shards, num_slices, X, y)

  locals().update(private_training_parameters)
  N = np.concatenate(X[0]).shape[0]
  delta = 1 / (N ** 1.1)
  epsilon = compute_eps_uniform(iterations, noise_multiplier, N, batch_size, delta)
  print(epsilon, delta)

  init_params, predict = models.conv()

  try:
    params = pickle.load(open('private_ensemble.pkl', 'rb'))
  except:
    print('Training full model (Shards={}, Slices={})...'.format(num_shards, num_slices))
    # params[0 <= i < num_shards][0 <= j <= num_slices] refers to the params trained on the first j slices of the i'th shard,
    # i.e., j == 0 yields randomly initialized params trained on no data, j == 1 yields params trained on the first slice, etc.
    params = get_trained_sharded_and_sliced_params(rng, init_params, predict, X, y, privately_train)
    pickle.dump(params, open('private_ensemble.pkl', 'wb'))

  targets = np.argmax(y_test, axis=1)
  predictions = sharded_and_sliced_predict(params, predict, X_test)
  print('Accuracy (ε = {:.4f}, δ = {}): {:.4}\n'.format(epsilon, delta, np.mean(predictions == targets)))

  print(get_votes(params, predict, X_test)[:20])

