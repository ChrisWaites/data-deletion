from jax import random
from .mnist import mnist

def shuffle(rng, *args):
  """Shuffles a set of args, each the same way."""
  return (random.permutation(rng, arg) for arg in args)

def get_dataset(rng, dataset):
  X_train, y_train, X_test, y_test = {
    'mnist': mnist,
  }[dataset]()

  # Shuffle training dataset
  X_train, y_train = shuffle(rng, X_train, y_train)

  return X_train, y_train, X_test, y_test
