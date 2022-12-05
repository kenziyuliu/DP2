import math
import numpy as np

import tensorflow_privacy as tfp


def compute_dp_sgd_privacy(n, batch_size, noise_mult, epochs, delta, drop_remainder=True):
  q = batch_size / n
  orders = ([1.1, 1.2, 1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75] +
            list(np.arange(5, 64, 0.5)) + [128, 256, 512])
  if drop_remainder:
    steps = n // batch_size * epochs
  else:
    # steps = int(math.ceil(epochs * n / batch_size))
    steps = epochs * int(math.ceil(n / batch_size))

  rdps = tfp.compute_rdp(q=q, noise_multiplier=noise_mult, steps=steps, orders=orders)
  eps, _, opt_order = tfp.get_privacy_spent(orders, rdps, target_delta=delta)
  return eps, opt_order


if __name__ == '__main__':
  import sys
  args = list(map(float, sys.argv[1:]))
  print(f'Args: (n, batch_size, noise_mult, epochs, delta) = {args}')
  eps, opt_order = compute_dp_sgd_privacy(*args, drop_remainder=False)
  print(f'opt_order: {opt_order}, eps = {eps:.5f}, delta = {args[-1]}')

