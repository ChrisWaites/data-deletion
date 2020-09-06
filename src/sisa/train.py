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
  private = False
  l2_norm_clip = 1.5
  noise_multiplier = 1.3
  iterations = 75
  batch_size = 32
  step_size = 0.15

  def loss(params, batch):
    inputs, targets = batch
    logits = predict(params, inputs)
    logits = stax.logsoftmax(logits)
    return -np.mean(np.sum(logits * targets, axis=1))

  def clipped_grad(params, l2_norm_clip, single_example_batch):
    grads = grad(loss)(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = np.linalg.norm([np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = np.max((total_grad_norm / l2_norm_clip, 1.))
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads)

  def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    batch = (np.expand_dims(batch[0], 1), np.expand_dims(batch[1], 1)) # Add batch dimension for when each example is separated
    clipped_grads = vmap(clipped_grad, (None, None, 0))(params, l2_norm_clip, batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [np.sum(g, 0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape) for r, g in zip(rngs, aggregated_clipped_grads)]
    normalized_noised_aggregated_clipped_grads = [g / batch_size for g in noised_aggregated_clipped_grads]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  @jit
  def private_update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)
    grads = private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, batch_size)
    return opt_update(i, grads, opt_state)

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
    if private:
      opt_state = private_update(temp, next(itercount), opt_state, next(batches))
    else:
      opt_state = update(temp, next(itercount), opt_state, next(batches))

  return get_params(opt_state)
