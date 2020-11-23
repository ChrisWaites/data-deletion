import matplotlib.pyplot as plt
from matplotlib import colors
from train import *
from numpy import genfromtxt
from sklearn import linear_model

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

  print(X_test)
  print(y_test)

  color_map = ['red', 'blue']
  plt.scatter(X_test[:, 0], X_test[:, 1], s=0.2, c=y_test, cmap=colors.ListedColormap(color_map))
  plt.savefig('plot.png', dpi=400)

  for iters in range(1, 6):
    model = linear_model.LogisticRegression(max_iter=iters)
    model.fit(X_test, y_test)
    y_hat = model.predict(X_test)
    print(np.mean(y_hat == y_test))
