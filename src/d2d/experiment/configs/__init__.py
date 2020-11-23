adult = {
  'alpha': 0.001,
  'l2_penalty': 0.05,
  'learning_rates': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
  'num_rounds': 30,
  'num_updates': 10,
  'perfect': False,
}

mnist = {
  'alpha': 9.0,
  'l2_penalty': 0.05,
  'learning_rates': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
  'num_rounds': 10,
  'num_updates': 10,
  'perfect': False,
}

moons = {
  'alpha': 7.0,
  'l2_penalty': 0.05,
  'learning_rates': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
  'num_rounds': 1,
  'num_updates': 3,
  'perfect': False,
}

projected_mnist = {
  'alpha': 9.0,
  'l2_penalty': 0.05,
  'learning_rates': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
  'num_rounds': 10,
  'num_updates': 10,
  'perfect': False,
}

gaussian = {
  'alpha': 7.0,
  'l2_penalty': 0.05,
  'learning_rates': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
  'num_rounds': 1,
  'num_updates': 3,
  'perfect': False,
}

gowalla = {
  'alpha': 9.0,
  'l2_penalty': 0.05,
  'learning_rates': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
  'num_rounds': 10,
  'num_updates': 10,
  'perfect': False,
}

def get_config(dataset):
  return {
    'adult': adult,
    'mnist': mnist,
    'moons': moons,
    'gaussian': gaussian,
    'projected_mnist': projected_mnist,
    'gowalla': gowalla,
  }[dataset]

