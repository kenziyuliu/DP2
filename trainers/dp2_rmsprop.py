from functools import partial

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import device_put, random, grad, jit, vmap, value_and_grad
from jax.example_libraries import optimizers

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils
from jax_utils import struct_add, struct_sub, struct_sqrt, struct_sq, struct_div, struct_add_scalar, struct_mul_scalar, struct_div_scalar
from jax_utils import struct_zeros_like, privatize_grad


class Trainer(BaseTrainer):
  """DP^2-RMSProp."""
  def __init__(self, args):
    super(Trainer, self).__init__(args)

  def train(self):
    def l2_loss_fn(params):
      return 0.5 * self.reg_lambda * jax_utils.global_l2_norm_sq(params)

    def init_state(key):
      # NOTE: For this variant, we want 2 LRs for the SGD and adaptive update
      # respectively; here we ignore `init_base_state` and maintain params directly.
      params = self.model.init(key, self.batch_template)
      print('[INFO] Num parameters:', jax_utils.num_params(params))
      A0 = struct_zeros_like(params)
      G0 = struct_zeros_like(params)
      count = 0
      return (params, A0, G0, count)

    @partial(jit, static_argnums=(0, ))
    def batch_update(branch, iter_idx, key, state, batch):
      """Currently proposed variant 2."""
      params, At, Gt, count = state
      key = random.fold_in(key, iter_idx)
      noise_mult = self.sigma
      # Data gradients. add fake batch dimension to data for vmapping
      vbatch = (batch[0][:, None], batch[1][:, None])  # new dim: (B, 1, ...).
      data_valgrad_fn = vmap(value_and_grad(self.data_loss_fn), in_axes=(None, 0), out_axes=0)
      example_losses, example_grads = data_valgrad_fn(params, vbatch)
      l2_grad = grad(l2_loss_fn)(params)

      ### Variant 2: branch 0 == DP-SGD, 1 == last iter of DP-SGD, 2 == Adaptive
      # Note that noising happens once at either branch == 0 or 2, so one `key` is enough.
      if branch <= 1:
        noisy_grad = privatize_grad(example_grads, key, self.clip1, noise_mult)
        Gt = struct_add(Gt, noisy_grad)  # G_t <- G_{t-1} + tilde(g_t)
        count = count + 1
        final_grad = noisy_grad
        final_lr = self.lr
      if branch == 1:
        Gt_avg_sq = struct_sq(struct_div_scalar(Gt, count))
        At = struct_add(struct_mul_scalar(At, self.rmsprop_gamma),
                        struct_mul_scalar(Gt_avg_sq, 1.0 - self.rmsprop_gamma))
        Gt = struct_zeros_like(params)
        count = 0
      if branch == 2:
        denom = struct_add_scalar(struct_sqrt(At), self.epsilon)
        precond_example_grads = struct_div(example_grads, denom)
        final_grad = privatize_grad(precond_example_grads, key, self.clip2, noise_mult)
        final_lr = self.lr2

      final_grad = struct_add(final_grad, l2_grad)
      params = struct_sub(params, struct_mul_scalar(final_grad, final_lr))

      # Also return batch metrics
      batch_loss = jnp.mean(example_losses)
      batch_acc = jnp.mean(
          self.pred_fn(params, batch[0]) == batch[1]) if self.is_classification else 0.0
      return batch_loss, batch_acc, key, (params, At, Gt, count)

    ## Training loop
    key = random.PRNGKey(self.seed)
    state = init_state(key)

    def branch_fn(i, interval=self.interval):
      # Decide which branch to go (for now let d1 == d2, i.e. DP-SGD and adaptive has same # iters).
      # 0 == DP-SGD, 1 == last iter of DP-SGD, 2 == adaptive.
      # e.g. returns [0, 0, 1, 2, 2, 2, 0, 0, 1, 2, 2, 2, ...] if interval == 3
      s1 = interval
      s2 = interval
      on_interval = (i+1+s2) % (s1+s2) == 0
      use_adaptive = (i // interval) % (1+1) > 0
      return 2 if use_adaptive else (1 if on_interval else 0)

    for epoch_idx in range(self.epochs):

      if epoch_idx % self.eval_every_epoch == 0:
        self.evaluate(epoch_idx, params=state[0])  # NOTE: Params is kept directly.

      for batch_idx, batch in enumerate(self.epoch_batches()):
        iter_idx = epoch_idx * self.epoch_steps + batch_idx
        batch = device_put(batch)  # Puts data to default device (gpu:0); done by default
        branch = branch_fn(iter_idx)
        batch_loss, batch_acc, key, state = batch_update(branch, iter_idx, key, state, batch)
        self.log_batch_metrics(iter_idx, batch_loss, batch_acc)

    return self.final_metrics()
