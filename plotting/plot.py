"""
General plotting function that takes
- a single run with repeats
- a single run without repeats
- a sweep (without repeats by default) with a specific hparam choice
and plots them allowing custom colors
"""
import argparse
import functools
import os
import pprint
import sys
from pathlib import Path
from typing import List
from glob import glob
from itertools import groupby

import numpy as np
from numpy import array  # For `eval` log files

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import plot_utils

# ------------ Plotting IMDB convergence ------------

# README examples on IMDB:
runs = [
    # baselines
    dict(runtype='single',
         method='sgd',
         label='DP-SGD',
         path='logs/imdb/dp_sgd_nm1.0/'),
    dict(runtype='single', method='rmsprop',
         label='DP-RMSProp',
         path='logs/imdb/dp_rmsprop_nm1.0/'),
    # ours
    dict(runtype='single',
         method='delay2',
         label='DP$^2$-RMSProp (ours)',
         path='logs/imdb/dp2_rmsprop_nm1.0'),
]
figsize = (4, 3)
y_factor = None
y_label = None

y_factor = y_factor or 1.0


def main(args):
  plot_utils.plt_setup(legendsize=args.legend, figsize=figsize, labelspacing=0.2)

  for run in runs:
    path, method, runtype = Path(run['path']), run['method'], run['runtype']
    print(f'Plotting {path=}')
    mean_batch, std_batch, mean_test, std_test = plot_utils.read_rundir(path,
                                                                        runtype,
                                                                        metric_name=args.metric)
    num_epochs = len(mean_test)
    mean_batch = mean_batch * y_factor
    mean_test = mean_test * y_factor

    if args.smooth_te:
      mean_test = plot_utils.smooth(mean_test, window_len=args.smooth_te)

    # Plot trajectory
    color = run['color'] if 'color' in run else plot_utils.get_color(method)
    plt.plot(mean_test, color=color, label=run['label'], linewidth=2)

    if args.std:
      xvals = list(range(num_epochs))
      plot_utils.plot_with_errorbar(xvals, mean_test, std_test, color=color)

    if args.train:
      mean_batch = mean_batch[::args.train_sample]
      mean_batch = plot_utils.smooth(mean_batch, window_len=args.smooth_tr)
      batch_xvals = np.linspace(0, num_epochs, num=len(mean_batch))  # Expand into epochs
      # plt.plot(batch_xvals, mean_batch, alpha=0.05, color=color)
      plt.plot(batch_xvals, mean_batch, ':', color=color)

  if args.title:
    plt.title(args.title)
  if args.grid:
    plt.grid(linestyle=':')
  if args.ylog:
    plt.yscale('log')
  if args.xlim:
    plt.xlim(args.xlim)
  if args.ylim:
    plt.ylim(args.ylim)

  y_label = 'MSE' if args.metric == 'loss' else args.metric
  plt.xlabel('\# epochs')
  plt.ylabel(f'test {y_label}')

  if args.legend:
    plt.legend(loc='best', frameon=True)

  plot_utils.save_fig(args.out)
  plt.show()


if __name__ == '__main__':
  args = plot_utils.arg_parser().parse_args()
  main(args)
