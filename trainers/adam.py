import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from jax import device_put, random, grad, jit, value_and_grad
from jax.example_libraries import optimizers

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils


class Trainer(BaseTrainer):
  """Non-private Adam baseline."""
  def __init__(self, args):
    super(Trainer, self).__init__(args)

  def train(self):
    # NOTE: Use `jax.example_libraries.optimizers` for a reliable baseline.
    adam_opt = optimizers.adam(self.lr, b1=self.beta1, b2=self.beta2, eps=self.epsilon)

    def loss_fn(params, batch):
      data_term = self.data_loss_fn(params, batch)
      l2_term = 0.5 * self.reg_lambda * jax_utils.global_l2_norm_sq(params)
      return data_term + l2_term

    def init_state(key):
      # Just ignore the base SGD optimizer since we dont manually transform the gradients.
      init_params = self.model.init(key, self.batch_template)
      return adam_opt.init_fn(init_params)

    @jit
    def batch_update(iter_idx, key, state, batch):
      key = random.fold_in(key, iter_idx)
      params = adam_opt.params_fn(state)
      batch_loss, final_grad = value_and_grad(loss_fn)(params, batch)
      batch_acc = jnp.mean(
          self.pred_fn(params, batch[0]) == batch[1]) if self.is_classification else 0.0
      return batch_loss, batch_acc, key, adam_opt.update_fn(iter_idx, final_grad, state)

    ## Training loop
    key = random.PRNGKey(self.seed)
    state = init_state(key)
    for epoch_idx in range(self.epochs):
      if epoch_idx % self.eval_every_epoch == 0:
        self.evaluate(epoch_idx, params=adam_opt.params_fn(state))

      for batch_idx, batch in enumerate(self.epoch_batches()):
        iter_idx = epoch_idx * self.epoch_steps + batch_idx
        batch = device_put(batch)  # Puts data to default device (gpu:0); done by default
        batch_loss, batch_acc, key, state = batch_update(iter_idx, key, state, batch)
        self.log_batch_metrics(iter_idx, batch_loss, batch_acc)

    return self.final_metrics()
