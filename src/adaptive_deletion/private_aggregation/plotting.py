
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
