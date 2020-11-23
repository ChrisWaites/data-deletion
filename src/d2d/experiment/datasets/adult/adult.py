import jax.numpy as np

from .clean_data import clean_adult_full
from sklearn import model_selection

def adult():
  X, y = clean_adult_full(scale_and_center=True, normalize=True, intercept=True, sampling_rate=0.1)

  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
  X_train, X_test, y_train, y_test = np.array(X_train.values), np.array(X_test.values), np.array(y_train.values), np.array(y_test.values)

  return X_train, y_train, X_test, y_test
