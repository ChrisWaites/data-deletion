from .adult import adult
from .mnist import mnist
from .projected_mnist import projected_mnist
from .gowalla import gowalla
from .gaussian import gaussian
from .moons import moons

def get_dataset(dataset):
  return {
    'adult': adult,
    'mnist': mnist,
    'projected_mnist': projected_mnist,
    'gowalla': gowalla,
    'gaussian': gaussian,
    'moons': moons,
  }[dataset]()
