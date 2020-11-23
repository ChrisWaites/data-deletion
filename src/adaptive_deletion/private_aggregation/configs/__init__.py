from .mnist import mnist

def get_config(dataset):
  return {
    'mnist': mnist,
  }[dataset]

