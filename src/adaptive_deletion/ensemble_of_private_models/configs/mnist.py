from jax.experimental import stax

def conv():
  init_fun, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(10),
  )
  def init_params(rng):
    return init_fun(rng, (-1, 28, 28, 1))[1]
  return init_params, predict

def feed_forward():
  init_fun, predict = stax.serial(
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(10),
  )
  def init_params(rng):
    return init_fun(rng, (-1, 28 * 28))[1]
  return init_params, predict

mnist = {
  'classifier': conv(),
}
