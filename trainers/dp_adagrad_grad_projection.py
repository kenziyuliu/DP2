import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import device_put, random, grad, jit, vmap, value_and_grad

from .trainer_base import BaseTrainer
from privacy_analysis.compute_privacy_sgm import compute_dp_sgd_privacy
import jax_utils
from jax_utils import struct_add, struct_sqrt, struct_sq, struct_div, struct_add_scalar, struct_mul_scalar, struct_zeros_like
from jax_utils import privatize_grad


class Trainer(BaseTrainer):
  """Implementation of KRRT'21 noisy AdaGrad https://arxiv.org/pdf/2008.06570.pdf.

  Simplifications:
  - We use diagonal preconditioner, so we wont need GOE noising on the matrix preconditioner
  - We do not have an "envelope of decaying sensitivity"
  """
  def __init__(self, args):
    super(Trainer, self).__init__(args)
    # Obtain a random batch of training data, and use that as the public data
    # Randomness comes from `loader`'s shuffle (depends on `seed`)
    print('[KRRT-noisy-adagrad] sampling training data as public data...')
    assert self.public_data_frac > 0.0
    self.train_loader, _, self.public_batch = self.loader_fn(batch_size=self.batch_size,
                                                             split='train',
                                                             is_eval=False,
                                                             seed=self.seed,
                                                             drop_remainder=True,
                                                             public_split=self.public_data_frac)
    self.epoch_steps = len(self.train_loader)  # need to update since this is used else where
    # for per-example ops: [ (B,1,...), (B,1) ]
    self.vpublic_batch = (self.public_batch[0][:, None], self.public_batch[1][:, None])

  def train(self):
    # def gaussian_orthonormal_ensemble(key, sigma, dim):
    #   """EXP: Generates a Gaussian orthonormal ensemble random matrix.
    #   Follows Def 8 of https://proceedings.mlr.press/v134/kairouz21a/kairouz21a.pdf.
    #   """
    #   mat = random.normal(key, shape=(dim, dim))
    #   mask = jnp.triu(jnp.ones((dim, dim)))
    #   mmat = mask * mat
    #   diag_indices = jnp.diag_indices_from(mmat)
    #   goe = mmat.at[diag_indices].multiply(jnp.sqrt(2))
    #   return goe

    def init_state(key):
      # Define the states for the current trainer.
      opt_state = self.init_base_state(key)
      params = self.sgd_opt.params_fn(opt_state)
      precond = struct_zeros_like(params)
      return (opt_state, precond)

    @jit
    def batch_update(iter_idx, key, state, batch):
      opt_state, precond = state
      key = random.fold_in(key, iter_idx)
      params = self.sgd_opt.params_fn(opt_state)
      # Data gradients. add fake batch dimension to data for vmapping
      vbatch = (batch[0][:, None], batch[1][:, None])  # new dim: (B, 1, ...).
      data_valgrad_fn = vmap(value_and_grad(self.data_loss_fn), in_axes=(None, 0), out_axes=0)
      example_losses, example_grads = data_valgrad_fn(params, vbatch)

      ########### EXP: KRRT-noisy-adagrad ###########
      dp_grad_key, dp_precond_key = random.split(key, 2)
      # Accumulate *clean* gradient into *diagonal* preconditioner
      clean_avg_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), example_grads)
      precond = struct_add(precond, struct_sq(clean_avg_grad))

      # Compute public projection using the current public gradient
      _, public_grads = data_valgrad_fn(params, self.vpublic_batch)  # Batched grads as PyTree
      vflatten_fn = vmap(jax_utils.struct_flatten, in_axes=0, out_axes=0)
      public_flat_grads = vflatten_fn(public_grads)  # (B, d)
      # Sum of per-example gradient outer products; follows https://arxiv.org/pdf/2007.03813.pdf
      batch_size = len(batch[0])  # shapes are static
      M_t = jnp.einsum('bd,be->de', public_flat_grads, public_flat_grads) / batch_size
      # EXPENSIVE: compute top-k eigvecs of M_t (shape dxd) as projection
      # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.eigh.html
      evals, evecs = jnp.linalg.eigh(M_t)
      topk_evecs = evecs[:, :self.grad_proj_k]  # (d, k)

      # Privatize preconditioner; notation follows Algorithm 1 of
      # https://proceedings.mlr.press/v134/kairouz21a/kairouz21a.pdf
      Gt = struct_sqrt(precond)
      clipped_Gt = jax_utils.global_l2_clip(Gt, self.clip2)
      flat_Gt = jax_utils.struct_flatten(clipped_Gt)
      flat_Gt = flat_Gt + self.clip2 * self.sigma * random.normal(dp_precond_key, flat_Gt.shape)
      # Do projection for the preconditioner:  (d, k) @ ((k, d) @ (d,)) = (d,)
      Ht = topk_evecs @ (topk_evecs.T @ flat_Gt)
      # Put preconditioner back into PyTree, and compute denominator (with epsilon)
      flat_denom = Ht + self.epsilon
      denom = jax_utils.struct_unflatten(flat_denom, precond)

      # Privatize clean gradient, and compute final preconditioned gradient
      noisy_grad = privatize_grad(example_grads, dp_grad_key, self.clip1, self.sigma)
      final_grad = struct_div(noisy_grad, denom)
      ################ End of KRRT-noisy-adagrad ################

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
