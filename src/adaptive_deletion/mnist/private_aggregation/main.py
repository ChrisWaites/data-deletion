import sys

sys.path.append('..')
sys.path.append('../..')

from sharding import *
import models
from train import train
import pickle
import matplotlib.pyplot as plt
import brewer2mpl

if __name__ == '__main__':
  rng = random.PRNGKey(0)
  num_shards, num_slices = 250, 1

  X, y, X_test, y_test = mnist()
  X, X_test = X.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)

  temp, rng = random.split(rng)
  X, y = shuffle(temp, X, y)

  print('X: {}, y: {}'.format(X.shape, y.shape))
  print('X_test: {}, y_test: {}'.format(X_test.shape, y_test.shape))

  # X[0<=i<num_shards][0<=j<num_slices] refers to the j'th slice of the i'th shard
  X, y = shard_and_slice(num_shards, num_slices, X, y)

  init_params, predict = models.conv()

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

  """
  # Plot histogram
  import matplotlib.pyplot as plt
  idx = np.argmax(y_test, 1).reshape(-1, 1)
  votes = np.take_along_axis(get_votes(params, predict, X_test), idx, 1).reshape(-1)
  counts = [0 for i in range(num_shards + 1)]
  for vote in votes:
    counts[int(vote.item())] += 1
  plt.hist(votes, bins=251)
  plt.savefig('251.png')
  plt.close('all')
  plt.hist(votes, bins=45)
  plt.savefig('45.png')
  plt.close('all')
  plt.hist(votes, bins=10)
  plt.savefig('10.png')
  plt.close('all')
  exit()
  """

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
  ax.set_xscale('log')

  ax.plot(
    per_example_epsilons,
    [nonprivate_accuracy for i in range(len(per_example_epsilons))],
    label='Non-private',
    color=next(colors),
    linestyle=next(linestyles),
  )

  for name, accs in zip(mechanism_names, mechanism_accs):
    ax.plot(
      per_example_epsilons,
      accs,
      label=name,
      color=next(colors),
      linestyle=next(linestyles),
    )

  ax.set_xlabel(r'$\varepsilon$')
  ax.set_ylabel(r'$\Pr[$Correct Classification$]$')
  ax.set_ylim(0.0, 1.0)

  ax.grid(True)
  ax.legend(loc='lower right', prop={'size': 6})
  ax.set_rasterized(True)

  fig.tight_layout()
  plt.savefig('plot.png', dpi=400)
