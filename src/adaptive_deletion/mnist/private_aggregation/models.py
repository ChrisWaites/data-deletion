from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh, Conv, MaxPool, Flatten

def conv():
  init_fun, predict = stax.serial(
    Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    Relu,
    MaxPool((2, 2), (1, 1)),
    Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    Relu,
    MaxPool((2, 2), (1, 1)),
    Flatten,
    Dense(32),
    Relu,
    Dense(10),
  )
  def init_params(rng):
    return init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def feed_forward():
  init_fun, predict = stax.serial(
    Dense(1024),
    Relu,
    Dense(1024),
    Relu,
    Dense(10),
  )
  def init_params(rng):
    return init_fun(rng, (-1, 28 * 28))[1]
  return init_params, predict


