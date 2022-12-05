import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def linear_model_fn(inputs, out_dim=1, zero_init=True, **kwargs):
  """Linear regression, logistic regression, SVM, etc."""
  w_init = hk.initializers.Constant(0) if zero_init else None
  return hk.Sequential([
      hk.Flatten(),
      # Linear models can (and often better to) be initialized as zero.
      hk.Linear(out_dim, w_init=w_init)
  ])(inputs)


def matfac_model_fn(inputs, matfac_dim_1, matfac_dim_2, embed_dim, **kwargs):
  """Matrix factorization, treating each element in data matrix as a record for DP.

  This impl performs **mini-batch private training** (so applicable to DP-SGD, etc.).
  The learnable parameters will be the low-rank matrices P and Q, where A ~= PQ
  with A being the data matrix. While we can compute matmul(P, Q) efficiently,
  we only compute the needed elements for a batch of coordnates (row, col) but
  still compute the gradients for the full P and Q matrices.

  Args:
    inputs: (batch_size, 2), where each row is a (row, col) pair.
    matfac_dim_1, matfac_dim_2 (int): the dimension of the data matrix; i.e. (matfac_dim_1, matfac_dim_2).
    embed_dim (int): the dimension of the low-rank matrices

  Returns:
    Predicted batch of coordinates with shape (batch_size,).
  """
  # Simply maintain two learnable low-rank matrices; follow initializer of hk.Embed.
  w1 = hk.get_parameter('w1',
                        shape=(matfac_dim_1, embed_dim),
                        init=hk.initializers.TruncatedNormal())
  w2 = hk.get_parameter('w2',
                        shape=(matfac_dim_2, embed_dim),
                        init=hk.initializers.TruncatedNormal())
  rows, cols = inputs[:, 0], inputs[:, 1]
  # Do row-wise and col-wise inner product to obtain the needed coordinates.``
  w1_rows = w1[rows, :]
  w2_cols = w2[cols, :]  # Transposed as part of `einsum`
  return jnp.einsum('nd,nd->n', w1_rows, w2_cols)  # (batch_size,)
