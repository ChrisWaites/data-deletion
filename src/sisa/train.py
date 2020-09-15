import jax.numpy as np
from jax import partial, grad, jit, random, vmap, pmap
from jax.tree_util import tree_flatten, tree_unflatten
from jax.experimental import optimizers, stax
import itertools

def train(rng, params, predict, X, y):
  """Generic train function called for each slice.

  Responsible for, given an rng key, a set of parameters to be trained, some inputs X and some outputs y,
  finetuning the params on X and y according to some internally defined training configuration.
  """
  iterations = 150
  batch_size = 64
  step_size = 0.001

  def loss(params, batch):
    inputs, targets = batch
    logits = predict(params, inputs)
    logits = stax.logsoftmax(logits)
    return -np.mean(np.sum(logits * targets, axis=1))

  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  def data_stream(rng):
    num_complete_batches, leftover = divmod(X.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)
    while True:
      temp, rng = random.split(rng)
      perm = random.permutation(temp, X.shape[0])
      for i in range(num_batches):
        batch_idx = perm[i*batch_size:(i+1)*batch_size]
        yield X[batch_idx], y[batch_idx]

  opt_init, opt_update, get_params = optimizers.adam(step_size)
  opt_state = opt_init(params)

  temp, rng = random.split(rng)
  batches = data_stream(rng)
  for i in range(iterations):
    temp, rng = random.split(rng)
    opt_state = update(temp, i, opt_state, next(batches))

  return get_params(opt_state)
