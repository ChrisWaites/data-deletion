import brewer2mpl
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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
