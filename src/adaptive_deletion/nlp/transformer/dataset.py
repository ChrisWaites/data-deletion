from typing import NamedTuple

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class LanguageDataset(NamedTuple):
  records: tf.data.Dataset
  vocab_size: int


def load(batch_size: int, sequence_length: int) -> LanguageDataset:
  """Load LM1B dataset, returning it and vocab_size."""
  ds, ds_info = tfds.load(
      'lm1b/subwords32k',
      split=tfds.Split.TRAIN,
      shuffle_files=True,
      with_info=True)

  crop_size = sequence_length + 1
  ds = ds.repeat()
  # Convert the dataset to constant-size int32 tensors.
  ds = ds.map(lambda d: tf.cast(d['text'], tf.int32))
  ds = ds.map(lambda t: _crop_or_pad(t, crop_size, pad_token=0))
  ds = ds.shuffle(batch_size * 10)
  # Create the language modeling observation/target pairs and batch them up.
  ds = ds.map(lambda t: dict(obs=t[:-1], target=t[1:]))
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = tfds.as_numpy(ds)
  return LanguageDataset(ds, ds_info.features['text'].encoder.vocab_size)


def _crop_or_pad(value, size, pad_token):
  """Either crop or pad value to be of size size."""
  val_size = tf.size(value)
  pad = lambda: tf.pad(value, [[0, size - val_size]], 'CONSTANT', constant_values=pad_token)
  return tf.cond(val_size < size, pad, lambda: value[:size])