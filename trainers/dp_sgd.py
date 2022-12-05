import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from jax import device_put, random, grad, jit, vmap, value_and_grad

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils


class Trainer(BaseTrainer):
  def __init__(self, args):
    super(Trainer, self).__init__(args)

  def train(self):
    def l2_loss_fn(params):
      return 0.5 * self.reg_lambda * jax_utils.global_l2_norm_sq(params)

    def init_state(key):
      # Define the states for the current trainer.
      return self.init_base_state(key)

    @jit
    def batch_update(iter_idx, key, state, batch):
      opt_state = state
      key = random.fold_in(key, iter_idx)
      params = self.sgd_opt.params_fn(opt_state)
      clip, noise_mult = self.clip1, self.sigma
      # Data gradients. Add fake batch dimension to data for vmapping.
      vbatch = (batch[0][:, None], batch[1][:, None])  # e.g. (B, 1, 28, 28, 1), (B, 1).
      # `vmap` the batch dimension.
      data_valgrad_fn = vmap(value_and_grad(self.data_loss_fn), in_axes=(None, 0), out_axes=0)
      # Loss: (B,); Grad: PyTree of (B, weights_dim) tensors.
      example_losses, example_grads = data_valgrad_fn(params, vbatch)
      noisy_grad = jax_utils.privatize_grad(example_grads, key, clip, noise_mult)
      # Add L2 reg to mean data grad (this shouldnt be clipped/noised).
      l2_grad = grad(l2_loss_fn)(params)
      final_grad = jax_utils.struct_add(noisy_grad, l2_grad)
      # Also return batch metrics
      batch_loss = jnp.mean(example_losses)
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
