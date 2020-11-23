import matplotlib.pyplot as plt
from matplotlib import colors
from train import *
from numpy import genfromtxt
from sklearn import linear_model
import warnings

def warn(*args, **kwargs):
  pass

warnings.warn = warn

if __name__ == '__main__':
  fe_params, lr_params = pickle.load(open('fe_params.pkl', 'rb')), pickle.load(open('lr_params.pkl', 'rb'))

  print(logistic_regression)
  # Example usage: logistic_regression(lr_params, X)

  print(feature_extractor)
  # Example usage: feature_extractor(fe_params, X)

  X_train = genfromtxt('X_train.csv', delimiter=',')
  y_train = genfromtxt('y_train.csv', delimiter=',')

  X_test = genfromtxt('X_test.csv', delimiter=',')
  y_test = genfromtxt('y_test.csv', delimiter=',')

  fig = plt.figure(figsize=(4, 3))
  ax = fig.add_subplot()

  ax.set_title('Projected MNIST Points')
  ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, linewidths=0, s=0.15, cmap=colors.ListedColormap(['Red', 'Blue']))

  fig.tight_layout()
  fig.savefig('points.png', dpi=400)
