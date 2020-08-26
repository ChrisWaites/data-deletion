import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax, Tanh

from mnist import mnist

from tqdm import tqdm
from time import time
import itertools


if __name__ == "__main__":
  def log_time(f):
    """Utility function for printing the execution time of a function in wall-time.
    """
    def g(*args):
      start = time()
      ret = f(*args)
      end = time()
      print('Function {} took {:.4} seconds.'.format(f.__name__, end - start))
      return ret
    return g

  def shuffle(rng, *args):
    """Shuffles a set of args, each the same way.
    """
    return (random.permutation(rng, arg) for arg in args)

  def shard_and_slice(num_shards, num_slices, *args):
    """Shards and slices an array.

    This means, after this function, an array X becomes indexable as
    X[0<=i<num_shards][0<=j<num_slices], referring to the j'th slice of the i'th shard.
    """
    return ([np.split(shard, num_slices) for shard in np.split(arg, num_shards)] for arg in args)

  rng = random.PRNGKey(0)
  num_shards, num_slices = 20, 5

  X, y, X_test, y_test = mnist()
  temp, rng = random.split(rng)
  X, y = shuffle(temp, X, y)

  # X[0<=i<num_shards][0<=j<num_slices] refers to the j'th slice of the i'th shard
  X, y = shard_and_slice(num_shards, num_slices, X, y)

  init_fun, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax
  )

  def init_params(rng):
    """Given an rng key, returns a randomly initialized set of params.
    """
    return init_fun(rng, (-1, 28 * 28))[1]

  def train(rng, params, X, y):
    """Generic train function.

    Responsible for, given an rng key, a set of parameters to be trained, some inputs X and some outputs y,
    finetuning the params on X and y according to some internally defined training configuration.
    """
    iterations = 15
    batch_size = 128
    step_size = 0.001

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)

    def loss(params, batch):
      inputs, targets = batch
      preds = predict(params, inputs)
      return -np.mean(np.sum(preds * targets, axis=1))

    @jit
    def update(i, opt_state, batch):
      params = get_params(opt_state)
      return opt_update(i, grad(loss)(params, batch), opt_state)

    def data_stream():
      num_train = len(X)
      num_complete_batches, leftover = divmod(num_train, batch_size)
      num_batches = num_complete_batches + bool(leftover)

      rng = random.PRNGKey(0)
      while True:
        temp, rng = random.split(rng)
        perm = random.permutation(temp, num_train)
        for i in range(num_batches):
          batch_idx = perm[i * batch_size:(i + 1) * batch_size]
          yield X[batch_idx], y[batch_idx]

    batches = data_stream()
    itercount = itertools.count()
    for _ in range(iterations):
      opt_state = update(next(itercount), opt_state, next(batches))

    return get_params(opt_state)

  @log_time
  def get_trained_sharded_and_sliced_params(rng, init_params, X, y, train):
    """Given a sharded and sliced dataset, constructs a sharded and sliced parameter object.
    """
    params = []
    for i in tqdm(range(len(X))):
      temp, rng = random.split(rng)
      shard_params = [init_params(temp)]
      for j in range(len(X[i])):
        X_train, y_train = np.concatenate(X[i][:j+1]), np.concatenate(y[i][:j+1])
        temp, rng = random.split(rng)
        finetuned_params = train(temp, shard_params[-1], X_train, y_train)
        shard_params.append(finetuned_params)
      params.append(shard_params)
    return params

  def sharded_and_sliced_predict(params, X):
    """Given a sharded and sliced dataset and set of params, defined the prediction function.
    """
    votes = np.zeros((X.shape[0], 10))
    for slice_params in params:
      predictions = predict(slice_params[-1], X)
      votes += np.eye(10)[np.argmax(predictions, axis=1)]
    return np.argmax(votes, axis=1)

  def get_location(idx, X):
    """Retrieves the location, i.e., the shard index i, slice index j, and value index k of the idx'th element in the struct.

    For example, given a sharded and sliced data object with two shards:
      [[_, _, _], [_, _, _], [_, #, _]], [...]
    Then, the # would be the 7th element (idx), and ti would be at location (shard: 0, slice: 2, value: 1).
    """
    num_examples = 0
    for i in range(len(X)):
      for j in range(len(X[i])):
        new_num_examples = num_examples + len(X[i][j])
        if idx < new_num_examples:
          return i, j, idx - num_examples
        num_examples = new_num_examples

  def delete_index(idx, *args):
    """Deletes the idx'th element of each arg (assumes they all have the same shape).
    """
    i, j, k = get_location(idx, args[0])
    for arg in args:
      arg[i][j] = arg[i][j][np.eye(len(arg[i][j]))[k] == 0.]
    return args

  @log_time
  def delete_and_retrain(rng, idx, params, X, y, train):
    """Deletes the idx'th element, and then retrains the sharded and sliced params accordingly.

    That is, if we want to delete the value at location (shard: i, slice: j, value: k), then we retrain
    all parameters of the i'th shard from slice j+1 onwards.
    """
    X, y = delete_index(idx, X, y)
    i, j, _ = get_location(idx, X)
    for s in range(j, len(X[i])):
      temp, rng = random.split(rng)
      X_train, y_train = np.concatenate(X[i][:s+1]), np.concatenate(y[i][:s+1])
      params[i][s+1] = train(temp, params[i][s], X_train, y_train)
    return params, X, y

  @log_time
  def delete_and_retrain_multiple(rng, idxs, params, X, y, train):
    """The same as delete_and_retrain, but allows for multiple indices to be specified.

    Can be more efficient than calling delete_and_retrain multiple times in sequence,
    because if two elements fall in the same slice, you don't repeat work.
    """
    num_examples = 0
    for i in tqdm(range(len(X))):
      update_occured = False
      for j in range(len(X[i])):
        new_num_examples = num_examples + len(X[i][j])
        mask = [True for i in range(len(X[i][j]))]
        while len(idxs) > 0 and idxs[0] < new_num_examples:
          update_occured = True
          idx = idxs.pop(0)
          mask[idx - num_examples] = False
        mask = np.array(mask)
        X[i][j], y[i][j] = X[i][j][mask], y[i][j][mask]
        if update_occured:
          temp, rng = random.split(rng)
          X_train, y_train = np.concatenate(X[i][:j+1]), np.concatenate(y[i][:j+1])
          params[i][j+1] = train(temp, params[i][j], X_train, y_train)
        num_examples = new_num_examples
    return params, X, y

  def total_examples(X):
    """Counts the total number of examples of a sharded and sliced data object X.
    """
    count = 0
    for i in range(len(X)):
      for j in range(len(X[i])):
        count += len(X[i][j])
    return count

  print('Training full model (Shards={}, Slices={})...'.format(num_shards, num_slices))
  # params[0 <= i < num_shards][0 <= j <= num_slices] refers to the params trained on the first j slices of the i'th shard,
  # i.e., j == 0 yields randomly initialized params trained on no data, j == 1 yields params trained on the first slice, etc.
  params = get_trained_sharded_and_sliced_params(rng, init_params, X, y, train)

  targets = np.argmax(y_test, axis=1)
  predictions = sharded_and_sliced_predict(params, X_test)
  print('Accuracy (N={}): {:.4}\n'.format(total_examples(X), np.mean(predictions == targets)))

  for delete_request in range(5):
    print('Deleting 1 datapoint...')
    temp, rng = random.split(rng)
    idx = random.randint(temp, (), 0, total_examples(X))
    temp, rng = random.split(rng)
    params, X, y = delete_and_retrain(temp, idx, params, X, y, train)
    predictions = sharded_and_sliced_predict(params, X_test)
    print('Accuracy (N={}): {:.4}\n'.format(total_examples(X), np.mean(predictions == targets)))

  num_points_to_delete = 10
  print('Deleting {} datapoint(s)...'.format(num_points_to_delete))
  idxs = sorted(list(random.randint(temp, (num_points_to_delete,), 0, total_examples(X))))
  temp, rng = random.split(rng)
  params, X, y = delete_and_retrain_multiple(temp, idxs, params, X, y, train)
  predictions = sharded_and_sliced_predict(params, X_test)
  print('Accuracy (N={}): {:.4}\n'.format(total_examples(X), np.mean(predictions == targets)))


