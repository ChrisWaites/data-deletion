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
  iterations = 75
  batch_size = 32
  step_size = 0.15

  def loss(params, batch):
    inputs, targets = batch
    logits = predict(params, inputs)
    logits = stax.logsoftmax(logits)
    return -np.mean(np.sum(logits * targets, axis=1))

  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  def data_stream():
    num_train = X.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    rng = random.PRNGKey(0)
    while True:
      temp, rng = random.split(rng)
      perm = random.permutation(temp, num_train)
      for i in range(num_batches):
        batch_idx = perm[i*batch_size:(i+1)*batch_size]
        yield X[batch_idx], y[batch_idx]

  opt_init, opt_update, get_params = optimizers.sgd(step_size)
  opt_state = opt_init(params)

  batches = data_stream()
  itercount = itertools.count()
  for _ in range(iterations):
    temp, rng = random.split(rng)
    opt_state = update(temp, next(itercount), opt_state, next(batches))

  return get_params(opt_state)
