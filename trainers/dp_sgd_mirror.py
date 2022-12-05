import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from jax import device_put, random, grad, jit, vmap, value_and_grad

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils
from jax_utils import struct_add, struct_sqrt, struct_sq, struct_div, struct_add_scalar, struct_mul_scalar, struct_zeros_like
from jax_utils import privatize_grad


class Trainer(BaseTrainer):
  """Implementation of https://arxiv.org/pdf/2112.00193.pdf
  Public Data-Assisted Mirror Descent for Private Model Training (PDA-DPMD)

  Notes:
  - We use euclidean distance as the Bregman divergence so mirror descent
    reduces to gradient descent
  - We evaluate the public gradient on a fixed batch of public data
  """
  def __init__(self, args):
    super(Trainer, self).__init__(args)
    # Obtain a random batch of training data, and use that as the public data
    # Randomness comes from `loader`'s shuffle (depends on `seed`)
    print('[PDA-DPMD] sampling training data as public data...')
    assert self.public_data_frac > 0.0
    self.train_loader, _, self.public_batch = self.loader_fn(
        batch_size=self.batch_size,
        split='train',
        is_eval=False,
        seed=self.seed,
        drop_remainder=True,
        public_split=self.public_data_frac)
    self.epoch_steps = len(self.train_loader)  # need to update since this is used else where
    self.total_iters = self.epoch_steps * self.epochs   # EXP: PDA-DPMD

  def train(self):
    def init_state(key):
      opt_state = self.init_base_state(key)
      return opt_state

    @jit
    def batch_update(iter_idx, key, state, batch):
      opt_state = state
      key = random.fold_in(key, iter_idx)
      params = self.sgd_opt.params_fn(opt_state)
      # Data gradients. add fake batch dimension to data for vmapping
      vbatch = (batch[0][:, None], batch[1][:, None])  # new dim: (B, 1, ...).
      data_valgrad_fn = vmap(value_and_grad(self.data_loss_fn), in_axes=(None, 0), out_axes=0)
      example_losses, example_grads = data_valgrad_fn(params, vbatch)
      private_grad = privatize_grad(example_grads, key, self.clip1, self.sigma)

      ########### EXP: PDA-DPMD ###########
      # Compute public batch gradient directly (no per-example operations)
      public_grad = grad(self.data_loss_fn)(params, self.public_batch)
      # Compute decaying interpolation factor
      alpha_t = jnp.cos(jnp.pi / 2.0 * iter_idx / self.total_iters)
      # Combine private / public gradients
      final_grad = struct_add(struct_mul_scalar(private_grad, alpha_t),
                              struct_mul_scalar(public_grad, 1.0 - alpha_t))
      ################ End of PDA-DPMD ################

      opt_state = self.sgd_opt.update_fn(iter_idx, final_grad, opt_state)
      # Also return batch metrics
      batch_loss = jnp.mean(example_losses)
      batch_acc = jnp.mean(
          self.pred_fn(params, batch[0]) == batch[1]) if self.is_classification else 0.0
      return batch_loss, batch_acc, key, opt_state

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
