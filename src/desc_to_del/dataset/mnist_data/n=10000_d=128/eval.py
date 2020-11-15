from train import *
from numpy import genfromtxt

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

  print(X_train)
  print(y_train)
