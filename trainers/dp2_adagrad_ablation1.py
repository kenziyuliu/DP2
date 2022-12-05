from functools import partial

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import device_put, random, grad, jit, vmap, value_and_grad

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils
from jax_utils import struct_add, struct_sqrt, struct_sq, struct_div, struct_add_scalar, struct_mul_scalar, struct_div_scalar
from jax_utils import struct_zeros_like, privatize_grad


class Trainer(BaseTrainer):
  """Ablation variant 1 for AdaGrad (see experiments section)."""
  def __init__(self, args):
    super(Trainer, self).__init__(args)

  def train(self):
    def l2_loss_fn(params):
      return 0.5 * self.reg_lambda * jax_utils.global_l2_norm_sq(params)

    def init_state(key):
      # Define the states for the current trainer.
      opt_state = self.init_base_state(key)
      params = self.sgd_opt.params_fn(opt_state)
      A0 = struct_zeros_like(params)
      G0 = struct_zeros_like(params)
      count = 0
      return (opt_state, A0, G0, count)

    @partial(jit, static_argnums=(0,))
    def batch_update(branch_st, iter_idx, key, state, batch):
      opt_state, At, Gt, count = state
      key = random.fold_in(key, iter_idx)
      params = self.sgd_opt.params_fn(opt_state)
      clip, noise_mult = self.clip1, self.sigma
      # Data gradients. add fake batch dimension to data for vmapping
      vbatch = (batch[0][:, None], batch[1][:, None])  # new dim: (B, 1, ...).
      data_valgrad_fn = vmap(value_and_grad(self.data_loss_fn), in_axes=(None, 0), out_axes=0)
      example_losses, example_grads = data_valgrad_fn(params, vbatch)
      noisy_grad = privatize_grad(example_grads, key, clip, noise_mult)
      l2_grad = grad(l2_loss_fn)(params)  # L2 is separate from private gradients

      ### Variant 1
      Gt = struct_add(Gt, noisy_grad)  # G_t <- G_{t-1} + tilde(g_t)
      count = count + 1

      if branch_st:
        # A <- A + (avg(G_t))^2
        At = struct_add(At, struct_sq(struct_div_scalar(Gt, count)))
        Gt = struct_zeros_like(params)
        count = 0
        final_grad = noisy_grad
        # print(f'[DEBUG] branch_st, iter_idx {iter_idx} '  # Comment away `jit` for printing
        #       f'count {count} '
        #       f'norm(noisy_grad) {jax_utils.global_l2_norm(noisy_grad):.4f} '
        #       f'norm(example_grads) {jax_utils.global_l2_norm(jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), example_grads)):.4f} '
        #       f'norm(Gt) {jax_utils.global_l2_norm(Gt):.4f} '
        #       f'norm(At) {jax_utils.global_l2_norm(At):.4f} '
        #       f'norm(final_grad) {jax_utils.global_l2_norm(final_grad):.4f}')
      else:
        denom = struct_add_scalar(struct_sqrt(At), self.epsilon)
        precond_example_grads = struct_div(example_grads, denom)
        key = random.fold_in(key, iter_idx)
        final_grad = privatize_grad(precond_example_grads, key, self.clip2, noise_mult)
        # print(f'[DEBUG] NOT branch_st, iter_idx {iter_idx} '  # Comment away `jit` for printing
        #       f'norm(noisy_grad) {jax_utils.global_l2_norm(noisy_grad):.4f} '
        #       f'norm(example_grads) {jax_utils.global_l2_norm(jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), example_grads)):.4f} '
        #       f'norm(denom) {jax_utils.global_l2_norm(denom):.4f} '
        #       f'norm(Gt) {jax_utils.global_l2_norm(Gt):.4f} '
        #       f'norm(At) {jax_utils.global_l2_norm(At):.4f} '
        #       f'norm(final_grad) {jax_utils.global_l2_norm(final_grad):.4f}')

      final_grad = struct_add(final_grad, l2_grad)
      opt_state = self.sgd_opt.update_fn(iter_idx, final_grad, opt_state)

      # Also return batch metrics
      batch_loss = jnp.mean(example_losses)
      batch_acc = jnp.mean(
          self.pred_fn(params, batch[0]) == batch[1]) if self.is_classification else 0.0
      return batch_loss, batch_acc, key, (opt_state, At, Gt, count)

    ## Training loop
    key = random.PRNGKey(self.seed)
    state = init_state(key)
    for epoch_idx in range(self.epochs):
      if epoch_idx % self.eval_every_epoch == 0:
        self.evaluate(epoch_idx, params=self.sgd_opt.params_fn(state[0]))

      for batch_idx, batch in enumerate(self.epoch_batches()):
        iter_idx = epoch_idx * self.epoch_steps + batch_idx
        batch = device_put(batch)  # Puts data to default device (gpu:0); done by default
        branch_st = (iter_idx % self.interval == 0)
        batch_loss, batch_acc, key, state = batch_update(branch_st, iter_idx, key, state, batch)
        self.log_batch_metrics(iter_idx, batch_loss, batch_acc)

    return self.final_metrics()
