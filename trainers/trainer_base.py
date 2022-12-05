import copy
import os
import sys
import functools
from pathlib import Path
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, device_put
from jax.example_libraries import optimizers
import haiku as hk
from tqdm import tqdm

import model_utils
import jax_utils
import data_utils
import utils


class BaseTrainer:
  def __init__(self, params):
    for key, val in params.items():
      setattr(self, key, val)

    if self.dataset == 'imdb':  # logistic regression
      data_dim = 10000
      self.model_fn = functools.partial(model_utils.linear_model_fn, out_dim=1)
      self.data_loss_fn = jax_utils.bce_loss
      self.pred_fn = jax_utils.logreg_classify
      self.delta = self.delta or 1e-5  # < 1/25000 (samples)
      self.loader_fn = data_utils.imdb_loader
      self.batch_template = np.zeros((self.batch_size, data_dim), dtype=np.float32)

    elif self.dataset == 'so_tag':  # multi-class classification
      data_dim = 10000
      # NOTE: do softmax and top-1 classification impl to match AdaDPS
      self.model_fn = functools.partial(model_utils.linear_model_fn, out_dim=500)
      self.data_loss_fn = jax_utils.sce_loss
      self.pred_fn = jax_utils.multiclass_classify
      self.delta = self.delta or 1e-6  # < 1/246092 (samples)
      if self.so_tag_cache:
        print('WARNING !!!! Caching the entire so_tag dataset in memory; (~30GB) per-run')
        self.loader_fn = data_utils.stackoverflow_tag_loader_cached
      else:
        self.loader_fn = data_utils.stackoverflow_tag_loader
      self.batch_template = np.zeros((self.batch_size, data_dim), dtype=np.float32)

    elif self.dataset == 'movielens':  # matrix factorization
      self.model_fn = functools.partial(model_utils.matfac_model_fn,
                                        matfac_dim_1=self.matfac_dim_1,
                                        matfac_dim_2=self.matfac_dim_2,
                                        embed_dim=self.matfac_embed_dim)
      self.data_loss_fn = functools.partial(jax_utils.mse_loss, squeeze=False)
      self.pred_fn = functools.partial(jax_utils.mse_loss, squeeze=False)
      self.delta = self.delta or 1 / ((self.matfac_dim_1 * self.matfac_dim_2) * self.matfac_density)
      self.loader_fn = functools.partial(data_utils.movielens_loader,
                                         matfac_dim_1=self.matfac_dim_1,
                                         matfac_dim_2=self.matfac_dim_2,
                                         density=self.matfac_density)
      # Batch is a list of coordinate (row, col) tuples
      self.batch_template = np.zeros((self.batch_size, 2), dtype=np.int32)

    else:
      raise NotImplementedError(f'Unsupported dataset: {self.dataset}')

    print(f'[INFO] Loading datasets...')
    self.train_loader, self.num_train_samples = self.loader_fn(self.batch_size,
                                                               split='train',
                                                               is_eval=False,
                                                               seed=self.seed,
                                                               drop_remainder=True)
    if self.train_epoch_eval:
      self.train_eval_loader, _ = self.loader_fn(self.test_batch_size,
                                                 seed=self.seed,
                                                 split='train',
                                                 is_eval=True)
    self.test_loader, self.num_test_samples = self.loader_fn(self.test_batch_size,
                                                             seed=self.seed,
                                                             split='test',
                                                             is_eval=True)
    self.epoch_steps = len(self.train_loader)
    print(f'[INFO] Train steps per epoch: {self.epoch_steps}, total epochs: {self.epochs}')

    ## Create model architecture & JIT-compile prediction/loss function
    self.model = hk.without_apply_rng(hk.transform(self.model_fn))
    self.params_template = self.model.init(jax.random.PRNGKey(0), self.batch_template)
    self.num_params = jax_utils.num_params(self.params_template)
    print('[INFO] Num parameters:', self.num_params)

    self.pred_fn = jit(functools.partial(self.pred_fn, self.model))
    self.data_loss_fn = jit(functools.partial(self.data_loss_fn, self.model))
    self.sgd_opt = optimizers.sgd(self.lr)  # Gives `init_fn`, `params_fn`, `update_fn`
    print('[INFO] Finished compiling loss/pred')

    ## Metrics
    self.batch_metrics = []
    self.train_epoch_metrics = []
    self.test_epoch_metrics = []
    self.train_start_time = time.perf_counter()

  def init_base_state(self, key):
    init_params = self.model.init(key, self.batch_template)
    opt_state = self.sgd_opt.init_fn(init_params)
    return opt_state

  def epoch_batches(self, no_bar=False):
    self.batches_bar = tqdm(self.train_loader.as_numpy_iterator(),
                            total=len(self.train_loader),
                            disable=no_bar or self.no_bar or self.is_parallel,
                            desc='[Train] ')
    return self.batches_bar

  def log_batch_metrics(self, iter_idx, batch_loss, batch_acc):
    if iter_idx % self.train_metrics_every_iter != 0:
      return

    batch_loss, batch_acc = float(batch_loss), float(batch_acc)
    # Save to file
    outdir = Path(self.outdir)
    utils.print_log(round(batch_loss, self.ndigits), fpath=outdir / f'batch_loss.txt')
    desc = f'[Train] Batch loss: {batch_loss:.4f}'
    metrics = [batch_loss]

    if self.is_classification:
      utils.print_log(round(batch_acc, self.ndigits), fpath=outdir / f'batch_acc.txt')
      desc += f', acc: {batch_acc:.4f}'
      metrics.append(batch_acc)

    self.batches_bar.set_description(desc)
    self.batch_metrics.append(metrics)

  def final_metrics(self):
    _batch, _train, _test = map(
        np.array, [self.batch_metrics, self.train_epoch_metrics, self.test_epoch_metrics])
    return _batch, _train, _test

  def train(self):
    raise NotImplementedError(f'BaseTrainer does not implement `train()`')

  def evaluate(self, epoch_idx, params, quiet=False):
    def _eval_dataset(params, split='train'):
      # Metrics are averaged over all batches / datapoints.
      # NOTE: only train/test split for now.
      loader = self.train_eval_loader if split == 'train' else self.test_loader
      loss, acc = 0, 0
      num_samples = 0
      for batch in loader.as_numpy_iterator():
        batch_x, batch_y = batch
        num_samples += len(batch_x)
        loss += jax.device_get(self.data_loss_fn(params, batch) * len(batch_x))
        if self.is_classification:
          acc += np.sum(self.pred_fn(params, batch_x) == batch_y)
      epoch_loss, epoch_acc = float(loss / num_samples), float(acc / num_samples)

      utils.print_log(round(epoch_loss, self.ndigits), fpath=outdir / f'{split}_epoch_loss.txt')

      if self.is_classification:
        utils.print_log(round(epoch_acc, self.ndigits), fpath=outdir / f'{split}_epoch_acc.txt')

      return epoch_loss, epoch_acc

    print('[Eval] ', end='')
    start_time = time.perf_counter()
    outdir = Path(self.outdir)

    test_loss, test_acc = _eval_dataset(params, split='test')
    test_metrics = [test_loss]
    metric_str = f'test loss: {test_loss:.5f}'

    if self.is_classification:
      metric_str += f', acc: {test_acc:.5f}'
      test_metrics.append(test_acc)

    self.test_epoch_metrics.append(test_metrics)

    # Display train metrics if available
    if self.train_epoch_eval:
      train_loss, train_acc = _eval_dataset(params, split='train')
      train_metrics = [train_loss]
      metric_str = f'train/test loss: {train_loss:.5f}/{test_loss:.5f}'

      if self.is_classification:
        metric_str += f', acc: {train_acc:.5f}/{test_acc:.5f}'
        train_metrics.append(train_acc)

      self.train_epoch_metrics.append(train_metrics)

    if not quiet:
      eval_duration = time.perf_counter() - start_time
      # train_duration = time.perf_counter() - self.train_start_time if epoch_idx > 0 else 0.0
      total_duration = time.perf_counter() - self.train_start_time
      train_duration = total_duration - eval_duration
      print(
          f'Run {self.run_idx} Epoch {epoch_idx} avg {metric_str} '
          f'(train/eval/total took {train_duration:.2f}/{eval_duration:.2f}/{total_duration:.2f}s)')
      self.train_start_time = time.perf_counter()
