from sisa import *
import models
from train import train

if __name__ == '__main__':
  rng = random.PRNGKey(0)
  num_shards, num_slices = 250, 1

  X, y, X_test, y_test = mnist()
  temp, rng = random.split(rng)
  X, y = shuffle(temp, X, y)

  print('X: {}, y: {}'.format(X.shape, y.shape))
  print('X_test: {}, y_test: {}'.format(X_test.shape, y_test.shape))

  # X[0<=i<num_shards][0<=j<num_slices] refers to the j'th slice of the i'th shard
  X, y = shard_and_slice(num_shards, num_slices, X, y)

  init_params, predict = models.feed_forward()

  print('Training full model (Shards={}, Slices={})...'.format(num_shards, num_slices))
  # params[0 <= i < num_shards][0 <= j <= num_slices] refers to the params trained on the first j slices of the i'th shard,
  # i.e., j == 0 yields randomly initialized params trained on no data, j == 1 yields params trained on the first slice, etc.
  params = get_trained_sharded_and_sliced_params(rng, init_params, predict, X, y, train)

  targets = np.argmax(y_test, axis=1)
  predictions = sharded_and_sliced_predict(params, predict, X_test)
  print('Accuracy (nonprivate): {:.4}\n'.format(np.mean(predictions == targets)))

  print(get_votes(params, X_test)[:20])

  per_example_epsilons = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 1.0]

  for mechanism in [exponential_mechanism, lnmax]:
    for per_example_epsilon in per_example_epsilons:
      temp, rng = random.split(rng)
      agg = lambda votes: exponential_mechanism(rng, votes, 1., per_example_epsilon)
      predictions = sharded_and_sliced_predict(params, predict, X_test, agg)
      print('Accuracy (eps={:.4}): {:.4}\n'.format(per_example_epsilon, np.mean(predictions == targets)))

