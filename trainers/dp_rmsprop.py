import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from jax import device_put, random, grad, jit, vmap, value_and_grad

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils
from jax_utils import struct_add, struct_sqrt, struct_sq, struct_div, struct_add_scalar, struct_mul_scalar


class Trainer(BaseTrainer):
  """Vanilla DP-RMSProp."""
  def __init__(self, args):
    super(Trainer, self).__init__(args)

  def train(self):
    def init_state(key):
      # Define the states for the current trainer.
      opt_state = self.init_base_state(key)
      params = self.sgd_opt.params_fn(opt_state)
      precond = jax_utils.struct_zeros_like(params)
      return (opt_state, precond)

    @jit
    def batch_update(iter_idx, key, state, batch):
      """Algorithm 2 in the current writeup."""
      opt_state, precond = state
      key = random.fold_in(key, iter_idx)
      params = self.sgd_opt.params_fn(opt_state)
      clip, noise_mult = self.clip1, self.sigma
      # Data gradients. add fake batch dimension to data for vmapping
      vbatch = (batch[0][:, None], batch[1][:, None])  # new dim: (B, 1, ...).
      data_valgrad_fn = vmap(value_and_grad(self.data_loss_fn), in_axes=(None, 0), out_axes=0)
      example_losses, example_grads = data_valgrad_fn(params, vbatch)
      noisy_grad = jax_utils.privatize_grad(example_grads, key, clip, noise_mult)

      ### DP-RMSProp ###
      precond = struct_add(struct_mul_scalar(precond, self.rmsprop_gamma),
                           struct_mul_scalar(struct_sq(noisy_grad), 1.0 - self.rmsprop_gamma))
      precond_sqrt = struct_add_scalar(struct_sqrt(precond), self.epsilon)
      final_grad = struct_div(noisy_grad, precond_sqrt)

      opt_state = self.sgd_opt.update_fn(iter_idx, final_grad, opt_state)

      # Also return batch metrics
      batch_loss = jnp.mean(example_losses)
      batch_acc = jnp.mean(
          self.pred_fn(params, batch[0]) == batch[1]) if self.is_classification else 0.0
      return batch_loss, batch_acc, key, (opt_state, precond)

    ## Training loop
    key = random.PRNGKey(self.seed)
    state = init_state(key)
    for epoch_idx in range(self.epochs):
      if epoch_idx % self.eval_every_epoch == 0:
        opt_state, _ = state
        self.evaluate(epoch_idx, params=self.sgd_opt.params_fn(opt_state))

      for batch_idx, batch in enumerate(self.epoch_batches()):
        iter_idx = epoch_idx * self.epoch_steps + batch_idx
        batch = device_put(batch)  # Puts data to default device (gpu:0); done by default
        batch_loss, batch_acc, key, state = batch_update(iter_idx, key, state, batch)
        self.log_batch_metrics(iter_idx, batch_loss, batch_acc)

    return self.final_metrics()
