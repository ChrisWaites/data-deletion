import numpy as np
import jax.numpy as jnp
from sklearn import preprocessing, cluster
import gzip

scaler = preprocessing.StandardScaler()

def gowalla():
  num_points = 500000
  num_train = int(0.9 * num_points)

  X = []
  with gzip.open('datasets/gowalla/loc-gowalla_totalCheckins.txt.gz', 'rt') as f:
    for line in list(f)[:num_points]:
      line = line.split('\t')
      lat, lon = float(line[2]), float(line[3])
      X.append((lat, lon))
  X = np.array(X).astype(np.float32)

  X_train = X[:num_train]
  X_test = X[num_train:]

  scaler = preprocessing.StandardScaler()

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(X)

  y_train = 2 * (kmeans.predict(X_train) - 0.5)
  y_test = 2 * (kmeans.predict(X_test) - 0.5)

  return jnp.array(X_train), jnp.array(y_train), jnp.array(X_test), jnp.array(y_test)

