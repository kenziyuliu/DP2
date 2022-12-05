import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from jax import device_put, random, grad, jit, value_and_grad

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils


class Trainer(BaseTrainer):
  """Non-private SGD baseline."""
  def __init__(self, args):
    super(Trainer, self).__init__(args)

  def train(self):
    def loss_fn(params, batch):
      data_term = self.data_loss_fn(params, batch)
      l2_term = 0.5 * self.reg_lambda * jax_utils.global_l2_norm_sq(params)
      return data_term + l2_term

    def init_state(key):
      # Define the states for the current trainer.
      return self.init_base_state(key)

    @jit
    def batch_update(iter_idx, key, state, batch):
      opt_state = state
      key = random.fold_in(key, iter_idx)
      params = self.sgd_opt.params_fn(opt_state)
      batch_loss, final_grad = value_and_grad(loss_fn)(params, batch)
      batch_acc = jnp.mean(
          self.pred_fn(params, batch[0]) == batch[1]) if self.is_classification else 0.0
      return batch_loss, batch_acc, key, self.sgd_opt.update_fn(iter_idx, final_grad, opt_state)

    ## Training loop
    key = random.PRNGKey(self.seed)
    state = init_state(key)
    for epoch_idx in range(self.epochs):
      if epoch_idx % self.eval_every_epoch == 0:
        self.evaluate(epoch_idx, params=self.sgd_opt.params_fn(state))

      for batch_idx, batch in enumerate(self.epoch_batches()):
        iter_idx = epoch_idx * self.epoch_steps + batch_idx
        batch = device_put(batch)  # Puts data to default device (gpu:0); done by default
        batch_loss, batch_acc, key, state = batch_update(iter_idx, key, state, batch)
        self.log_batch_metrics(iter_idx, batch_loss, batch_acc)

    return self.final_metrics()
