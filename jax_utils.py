import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, device_put
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.scipy.special import expit as sigmoid
import optax
import haiku as hk
import numpy as np

######################################
############## DP utils ##############
######################################


def global_l2_clip(tensor_struct, clip: float):
  t_list, tree_def = tree_flatten(tensor_struct)
  tensor_norms = jnp.asarray([jnp.linalg.norm(t.reshape(-1), ord=2) for t in t_list])
  global_norm = jnp.linalg.norm(tensor_norms)
  norm_factor = jnp.minimum(clip / (global_norm + 1e-15), 1.0)
  clipped_t_list = [t * norm_factor for t in t_list]
  return tree_unflatten(tree_def, clipped_t_list)


def privatize_grad(example_grads, key, clip, noise_mult):
  """Privatize a batch of per-example gradients (in the form of PyTree)."""
  # Per-example gradient clipping
  clip_fn = vmap(global_l2_clip, in_axes=(0, None), out_axes=0)
  example_grads = clip_fn(example_grads, clip)  # PyTree of (B, weight_dim)
  # Sum
  flat_example_grads, tree_def = tree_flatten(example_grads)  # List of (B, weight_dim)
  batch_size = len(flat_example_grads[0])  # 1st dim of per-example grad tensors
  flat_sum_grads = [g.sum(axis=0) for g in flat_example_grads]
  # Noise & mean
  keys = random.split(key, len(flat_sum_grads))
  flat_mean_noisy_grads = [(g + clip * noise_mult * random.normal(k, g.shape)) / batch_size
                           for k, g in zip(keys, flat_sum_grads)]
  mean_noisy_grads = tree_unflatten(tree_def, flat_mean_noisy_grads)
  return mean_noisy_grads


############################################
############## Training utils ##############
############################################


def hinge_loss(model, params, batch, reg=0.1):
  inputs, targets = batch  # (n, ...), (n,)
  param_vec = struct_flatten(params)
  # (b, 1) -> (b,) if last dimension is 1
  preds = model.apply(params=params, inputs=inputs).squeeze()
  per_example_losses = jax.nn.relu(1.0 - targets * preds) + 0.5 * reg * (param_vec @ param_vec)
  return jnp.mean(per_example_losses)


def bce_loss(model, params, batch):
  inputs, targets = batch
  logits = model.apply(params=params, inputs=inputs).squeeze()
  per_example_losses = optax.sigmoid_binary_cross_entropy(logits, targets)
  return jnp.mean(per_example_losses)


def sce_loss(model, params, batch):
  inputs, targets = batch
  logits = model.apply(params=params, inputs=inputs)  # (B, C) or (B, T, C)
  targets = jax.nn.one_hot(targets, logits.shape[-1])  # (B, C) or (B, T, C)
  per_example_losses = optax.softmax_cross_entropy(logits, targets)  # (B,) or (B, T)
  return jnp.mean(per_example_losses)  # avg away all dims.


def mse_loss(model, params, batch, squeeze=True):
  inputs, targets = batch
  preds = model.apply(params=params, inputs=inputs)
  if squeeze:
    preds = preds.squeeze()
  per_example_losses = 0.5 * (preds - targets)**2
  return jnp.mean(per_example_losses)


def multiclass_classify(model, params, batch_inputs):
  logits = model.apply(params=params, inputs=batch_inputs)  # (B, C) or (B, T, C)
  pred_class = jnp.argmax(logits, axis=-1)  # (B,) or (B, T)
  return pred_class


def linear_svm_classify(model, params, batch_inputs):
  preds = model.apply(params=params, inputs=batch_inputs).squeeze()
  return jnp.sign(preds)


def logreg_classify(model, params, batch_inputs, temperature=1.0):
  preds = model.apply(params=params, inputs=batch_inputs).squeeze()
  preds = sigmoid(preds / temperature)
  return jnp.round(preds)  # (n_samples,)


def regression_pred(model, params, batch_inputs):
  return model.apply(params=params, inputs=batch_inputs).squeeze()


def global_l2_norm_sq(tensor_struct):
  """Computes the global L2 norm squared in an autodiff friendly way.

  You can get NaNs from `jnp.linalg.norm`; the gist is that `norm` is not
  differentiable at 0, but `squared-norm` is indeed differentiable at 0
  (needed for l2 regularization). See:
    https://github.com/google/jax/issues/3058
    https://github.com/google/jax/issues/6484
  """
  flat_vec = struct_flatten(tensor_struct)
  return flat_vec @ flat_vec


def global_l2_norm(tensor_struct):
  """Computes the global L2 norm of tensor tree."""
  return jnp.sqrt(global_l2_norm_sq(tensor_struct))


##########################################
############## Struct utils ##############
##########################################


def num_params(struct):
  param_list, _ = tree_flatten(struct)
  return np.sum([w.size for w in param_list])  # Use numpy since shape is static.


@jit
def struct_add(struct_1, struct_2):
  return tree_map(jnp.add, struct_1, struct_2)


@jit
def struct_sub(struct_1, struct_2):
  return tree_map(jnp.subtract, struct_1, struct_2)


@jit
def struct_mul(struct_1, struct_2):
  return tree_map(jnp.multiply, struct_1, struct_2)


@jit
def struct_div(struct_1, struct_2):
  return tree_map(jnp.divide, struct_1, struct_2)


@jit
def struct_sq(struct):
  return tree_map(jnp.square, struct)


@jit
def struct_sqrt(struct):
  return tree_map(jnp.sqrt, struct)


@jit
def struct_add_scalar(struct, value):
  t_list, tree_def = tree_flatten(struct)
  new_t_list = [t + value for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


@jit
def struct_mul_scalar(struct, factor):
  t_list, tree_def = tree_flatten(struct)
  new_t_list = [t * factor for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


@jit
def struct_div_scalar(struct, factor):
  return struct_mul_scalar(struct, 1.0 / factor)


@jit
def struct_average(struct_list, weights=None):
  def average_fn(*tensor_list):
    return jnp.average(jnp.asarray(tensor_list), axis=0, weights=weights)

  return tree_map(average_fn, *struct_list)


@jit
def struct_flatten(struct):
  tensors, tree_def = tree_flatten(struct)
  flat_vec = jnp.concatenate([t.reshape(-1) for t in tensors])
  return flat_vec


@jit
def struct_unflatten(flat_vec, struct_template):
  t_list, tree_def = tree_flatten(struct_template)
  pointer, split_list = 0, []
  for tensor in t_list:
    length = np.prod(tensor.shape)  # Shape is static so np is fine
    split_vec = flat_vec[pointer:pointer + length]
    split_list.append(split_vec.reshape(tensor.shape))
    pointer += length
  return tree_unflatten(tree_def, split_list)


@jit
def struct_concat(struct_list):
  flat_vecs = [struct_flatten(struct) for struct in struct_list]
  return jnp.concatenate(flat_vecs)


@jit
def struct_zeros_like(struct):
  return tree_map(jnp.zeros_like, struct)
