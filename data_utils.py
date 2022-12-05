import csv
import collections
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp

import tensorflow as tf
# We only use TF for data loading; ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')
import tensorflow_datasets as tfds

#############################
### Dataset loading utils ###
#############################


def imdb_loader(batch_size,
                split,
                data_dir='data/imdb',
                is_eval=False,
                seed=None,
                drop_remainder=True,
                public_split=0.0):
  """IMDB logistic regression dataset. x_train/x_test: (25000, 10000).

  Args:
    batch_size: batch size for the loader
    split: 'train' or 'test' sets
    data_dir: directory to load the dataset; contains `.npy` files
    is_eval: whether this loader is for evaluation
    seed: random seed for shuffling
    drop_remainder: whether to drop the last batch if it is smaller than `batch_size`
    public_split: fraction of data to use as "public data"; if >0, the public split
      is returned along the data loader

  Returns:
    The data loader, number of examples, and (if `public_split` > 0) the public data.
  """
  # IMDB is not too large so we load all into memory directly for speed.
  # We may want to allow a `mmap` mode for instances with limited RAM.
  assert split in ('train', 'test'), f'{split=} not supported!'
  fname_x, fname_y = f'x_{split}.npy', f'y_{split}.npy'
  dir_path = Path(data_dir)
  data_x = np.load(dir_path / fname_x)
  data_y = np.load(dir_path / fname_y)
  assert len(data_x) == len(data_y)
  num_examples = len(data_x)

  if (not is_eval) or public_split:
    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(len(data_y))
    data_x, data_y = data_x[train_perm], data_y[train_perm]

  if public_split:
    num_public_samples = int(public_split * num_examples)
    num_examples -= num_public_samples
    public_x, public_y = data_x[:num_public_samples], data_y[:num_public_samples]
    data_x, data_y = data_x[num_public_samples:], data_y[num_public_samples:]

  print(f'[INFO] IMDB binary classification {split=} {num_examples=} ({public_split=})')

  if is_eval:
    loader = _create_tf_loader(data_x, data_y, batch_size, shuffle_size=None, drop_remainder=False)
  else:
    loader = _create_tf_loader(
        data_x,
        data_y,
        batch_size,
        shuffle_size=num_examples,  # Full training set
        seed=seed,
        drop_remainder=drop_remainder)

  if public_split:
    return loader, num_examples, (public_x, public_y)

  return loader, num_examples


def stackoverflow_tag_loader(batch_size,
                             split,
                             data_dir='data/stackoverflow_tag',
                             is_eval=False,
                             seed=None,
                             drop_remainder=True,
                             public_split=0.0):
  """Loads (centralized, subsampled) StackOverflow tag prediction dataset with generators.

  See `imdb_loader` for parameter descriptions.
  """
  assert split in ('train', 'test')
  fname_x, fname_y = f'x_{split}.npy', f'y_{split}.npy'
  dir_path = Path(data_dir)
  rng = np.random.default_rng(seed)

  # Use mmap & generator to prevent memory overload since stackoverflow is large
  data_x = np.load(dir_path / fname_x, mmap_mode='r')
  data_y = np.load(dir_path / fname_y)
  assert len(data_x) == len(data_y)
  num_examples = len(data_x)
  indices = rng.permutation(num_examples)

  if public_split:
    num_public_samples = int(public_split * num_examples)
    num_examples -= num_public_samples
    indices, public_indices = indices[num_public_samples:], indices[:num_public_samples]
    # NOTE: below assumes `public_indices` is a small fraction since `data_x` is memmap'ed
    public_x, public_y = data_x[public_indices], data_y[public_indices]

  print(f'[INFO] StackOverflow Tag Prediction ({split}) {num_examples} samples ({public_split=})')

  def data_gen():
    # Shuffle for non-public indices
    nonlocal indices
    indices = rng.permutation(indices)
    for i in indices:
      yield (data_x[i], data_y[i])

  loader = tf.data.Dataset.from_generator(data_gen,
                                          output_types=(tf.float32, tf.int64),
                                          output_shapes=((10000, ), ()))
  # Manually set the length of the dataset
  loader = loader.apply(tf.data.experimental.assert_cardinality(num_examples))

  if is_eval:
    loader = _config_tf_loader(loader, batch_size, shuffle_size=None, drop_remainder=False)
  else:
    # NOTE: the shuffle size determines memory usage
    loader = _config_tf_loader(
        loader,
        batch_size,
        shuffle_size=10000,  # around 4% of training set (total 246092 examples)
        seed=seed,
        drop_remainder=drop_remainder)

  if public_split:
    return loader, num_examples, (public_x, public_y)

  return loader, num_examples


def stackoverflow_tag_loader_cached(batch_size,
                                    split,
                                    data_dir='data/stackoverflow_tag',
                                    is_eval=False,
                                    seed=None,
                                    drop_remainder=True,
                                    public_split=0.0):
  """The same SO tag loader, but load the dataset in memory (typically >30GB)!"""
  assert split in ('train', 'test'), f'{split=} not supported!'
  fname_x, fname_y = f'x_{split}.npy', f'y_{split}.npy'
  dir_path = Path(data_dir)
  data_x = np.load(dir_path / fname_x)
  data_y = np.load(dir_path / fname_y)
  assert len(data_x) == len(data_y)
  num_examples = len(data_x)

  if (not is_eval) or public_split:
    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(len(data_y))
    data_x, data_y = data_x[train_perm], data_y[train_perm]

  if public_split:
    num_public_samples = int(public_split * num_examples)
    num_examples -= num_public_samples
    public_x, public_y = data_x[:num_public_samples], data_y[:num_public_samples]
    data_x, data_y = data_x[num_public_samples:], data_y[num_public_samples:]

  print(f'[INFO] StackOverflow Tag Prediction ({split}) {num_examples} samples ({public_split=})')

  if is_eval:
    loader = _create_tf_loader(data_x,
                               data_y,
                               batch_size,
                               shuffle_size=None,
                               drop_remainder=False,
                               cache=True)
  else:
    loader = _create_tf_loader(
        data_x,
        data_y,
        batch_size,
        shuffle_size=10000,  # around 4% of training set (total 246092 examples)
        seed=seed,
        drop_remainder=drop_remainder,
        cache=True)

  if public_split:
    return loader, num_examples, (public_x, public_y)

  return loader, num_examples


def _load_movielens_100k(data_dir):
  names = ['user_id', 'item_id', 'rating', 'timestamp']
  ratings_df = pd.read_csv(Path(data_dir) / 'u.data', sep='\t', names=names)
  X = ratings_df[['user_id', 'item_id']].values
  y = ratings_df['rating'].values
  row_indices = X[:, 0]
  col_indices = X[:, 1]
  values = y.astype(np.float32)
  n_users = len(ratings_df['user_id'].unique())
  n_items = len(ratings_df['item_id'].unique())
  shape = (n_users + 1, n_items + 1)
  # print(shape)
  # print(len(row_indices), len(col_indices), len(y))
  return shape, row_indices, col_indices, values


def movielens_loader(batch_size,
                     matfac_dim_1: int,
                     matfac_dim_2: int,
                     density: float,
                     seed=None,
                     split=None,
                     data_dir='data/movielens_100k',
                     is_eval=False,
                     drop_remainder=True,
                     public_split=0.0):
  """Loader for MovieLens matrix factorization task."""
  if public_split:
    raise NotImplementedError('`public_split` not implemented for MovieLens')

  assert split in ('train', 'test'), f'{split=} not supported!'
  shape, rows, cols, vals = _load_movielens_100k(data_dir)

  # Create matrix that can be indexed by (row, col) pairs
  mat = sp.sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
  # Create loader that randomly samples the non-zero coordinates.
  # This implies that the "dataset" is the set of non-zero coordinates onto
  # which we apply example-level DP.
  data_x = np.array([rows, cols]).T  # (n_nonzero, 2)
  data_y = vals  # (n_nonzero,)
  num_examples = len(data_x)
  assert len(data_x) == len(data_y)
  density = len(vals) / (shape[0] * shape[1])

  rng = np.random.default_rng(seed)
  perm = rng.permutation(num_examples)
  data_x, data_y = data_x[perm], data_y[perm]
  train_num_examples = int(num_examples * 0.8)

  if split == 'test':
    data_x, data_y = data_x[train_num_examples:], data_y[train_num_examples:]
  else:
    data_x, data_y = data_x[:train_num_examples], data_y[:train_num_examples],

  split_examples = len(data_x)

  print(f'[INFO] MovieLens_100k has {num_examples} non-zero coords in total '
        f'with {split_examples} in {split=} in a {shape} matrix of {density=} ({public_split=})')

  if is_eval:
    loader = _create_tf_loader(data_x, data_y, batch_size, shuffle_size=None, drop_remainder=False)
  else:
    loader = _create_tf_loader(
        data_x,
        data_y,
        batch_size,
        shuffle_size=split_examples,  # Shuffle all data
        seed=seed,
        drop_remainder=drop_remainder)

  return loader, split_examples


###################################
### TensorFlow dataloading for JAX
### From https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html
###################################


def _config_tf_loader(loader, batch_size, shuffle_size=50000, seed=None, drop_remainder=True):
  if shuffle_size is not None:
    # A fixed `seed` still allows every new epoch to have different shuffle.
    loader = loader.shuffle(shuffle_size, seed=seed, reshuffle_each_iteration=True)
  loader = loader.batch(batch_size, drop_remainder=drop_remainder)
  loader = loader.prefetch(tf.data.AUTOTUNE)
  return loader


def _create_tf_loader(x_data: np.ndarray,
                      y_data: np.ndarray,
                      batch_size,
                      shuffle_size=50000,
                      seed=None,
                      drop_remainder=True,
                      cache=False):
  """Constructs an iterable of numpy batches with tf.data (more CPU efficient)."""
  # See also https://www.tensorflow.org/datasets/performances.
  loader = tf.data.Dataset.from_tensor_slices((x_data, y_data))
  if batch_size == -1:
    batch_size = len(x_data)  # Full batch.
  if cache:
    loader = loader.cache()
  return _config_tf_loader(loader, batch_size, shuffle_size, seed, drop_remainder)


def benchmark(dataset, num_epochs=2):
  start_time = time.perf_counter()
  for epoch_num in range(num_epochs):
    for i, sample in enumerate(dataset.as_numpy_iterator()):
      # time.sleep(0.001)  # Performing a training step
      if i == 0:
        print(sample[1])
  print('Execution time:', time.perf_counter() - start_time)
