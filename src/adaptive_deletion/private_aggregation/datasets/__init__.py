from .mnist import mnist

def get_dataset(dataset):
  return {
    'mnist': mnist,
  }[dataset]()

