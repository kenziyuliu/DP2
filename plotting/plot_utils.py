import argparse
import functools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import array  # for `eval`
from scipy.signal import savgol_filter


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--out', type=str)
  parser.add_argument('--title', type=str, help='title')
  parser.add_argument('--metric', type=str, choices=('loss', 'acc', 'ppl'), default='acc')
  parser.add_argument('--xlim', nargs='+', type=float, help='x-axis limits')
  parser.add_argument('--ylim', nargs='+', type=float, help='y-axis limits')
  parser.add_argument('--delay_lower', type=int, help='take dirs with delay >= this')
  parser.add_argument('--delay_upper', type=int, help='take dirs with delay <= this')
  parser.add_argument('--smooth_tr', type=int, default=101, help='window for smoothing')
  parser.add_argument('--smooth_te', type=int, default=None, help='window for smoothing')
  parser.add_argument('--legend', type=float, default=10, help='legend size')
  parser.add_argument('--legend_out', action='store_true', help='plot legend outside of plot')
  parser.add_argument('--std', action='store_true', help='plot std of test')
  parser.add_argument('--grid', action='store_true', help='plot grid')
  parser.add_argument('--train', action='store_true', help='plot train batch metrics')
  parser.add_argument('--train_sample', type=int, default=1, help='sample train batch metrics every iter')
  parser.add_argument('--no_base', action='store_true', help='skip baselines')
  parser.add_argument('--no_show', action='store_true', help='do not plt.show()')
  parser.add_argument('--xlog', action='store_true', help='log scale for x-axis')
  parser.add_argument('--ylog', action='store_true', help='log scale for y-axis')
  parser.add_argument('--private_only', action='store_true', help='try not to show non-private results')
  parser.add_argument('--dataset', type=str, choices=('imdb', 'so_tag', 'movielens'), default='imdb')
  parser.add_argument('--figsize', nargs='+', help='figure size')
  return parser


def plt_setup(legendsize=12,
              figsize=(5, 4),
              labelspacing=0.3,
              tick_size=12,
              axes_size=13,
              markersize=5):
  matplotlib.rcParams['font.family'] = "sans-serif"
  matplotlib.rcParams['font.sans-serif'] = "Arial"
  matplotlib.rc('text', usetex=True)
  if markersize:
    matplotlib.rcParams['lines.markersize'] = markersize

  plt.rc('font', size=14)  # controls default text sizes
  plt.rc('axes', titlesize=16)  # fontsize of the axes title
  plt.rc('axes', labelsize=axes_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=tick_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=tick_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=legendsize)  # legend fontsize
  if labelspacing:
    plt.rc('legend', labelspacing=labelspacing)
  # plt.rc('figure', titlesize=titlesize)  # fontsize of the figure title
  plt.rc('figure', figsize=figsize)


@functools.cache
def read_single_col(filename):
  """Assumes a single column of numbers."""
  with open(filename) as f:
    return np.array([eval(line) for line in f.readlines()])


@functools.cache
def read_rundir_final(path: Path, runtype: str, metric_name: str = 'acc'):
  """
  For a given `rundir` containing results, return the batch metrics and
  the test epoch metrics as a tuple of lists.

  Returns: a tuple of 4 numbers: (final_batch_mean, final_batch_std, final_test_mean, final_test_std)
  """
  assert runtype in ('rep', 'single', 'sweep')
  metric_names = ['loss', 'acc', 'ppl']  # HACK: assumes metrics are in this order

  # Summarize final results by taking last window avg
  last_avg = lambda x, window: (x[-window:].mean(axis=0).astype(float) if len(x) else 0)

  if runtype == 'rep':
    fnames = ['batch', 'test']
    fnames = [f'summary_final_{fname}_metrics.txt' for fname in fnames]
    mean, std = [], []
    for fname in fnames:
      with open(path / fname) as f:
        final_dict = eval(f.read())
        mean.append(final_dict['mean'][metric_names.index(metric_name)])
        std.append(final_dict['std'][metric_names.index(metric_name)])
    mean_batch, mean_test = mean
    std_batch, std_test = std

  elif runtype == 'sweep':
    best_dir, best_result_vals = read_best_dir_from_sweep(path)
    mean_batch = read_single_col(Path(best_dir) / f'batch_{metric_name}.txt')
    mean_test = read_single_col(Path(best_dir) / f'test_epoch_{metric_name}.txt')
    mean_batch = last_avg(mean_batch, window=50)
    mean_test = last_avg(mean_test, window=4)
    std_batch, std_test = 0, 0

  else:
    mean_batch = read_single_col(path / f'batch_{metric_name}.txt')
    mean_test = read_single_col(path / f'test_epoch_{metric_name}.txt')
    mean_batch = last_avg(mean_batch, window=50)
    mean_test = last_avg(mean_test, window=4)
    std_batch, std_test = 0, 0

  # (T, <num_metrics>) each --> (T,) each
  print(f'read_rundir_final: {mean_batch=}, {std_batch=}, {mean_test=}, {std_test=}')
  return mean_batch, std_batch, mean_test, std_test


@functools.cache
def read_rundir(path: Path, runtype: str, metric_name: str = 'acc'):
  """
  For a given `rundir` containing results, return the batch metrics and
  the test epoch metrics as a tuple of lists.

  Returns: a tuple of 4 (T,) arrays, for the specific metric
  """
  assert runtype in ('rep', 'single', 'sweep')
  metric_names = ['loss', 'acc', 'ppl']  # HACK: assumes metrics are in this order
  if runtype == 'rep':
    fnames = ['mean_batch', 'std_batch', 'mean_test', 'std_test']
    fnames = [f'{fname}_metrics.txt' for fname in fnames]
    metric_list = []
    for fname in fnames:
      with open(path / fname) as f:
        metric_vals = np.array(eval(f.read()))
        metric_list.append(metric_vals[:, metric_names.index(metric_name)])
    mean_batch, std_batch, mean_test, std_test = metric_list

  elif runtype == 'sweep':
    best_dir, best_result_vals = read_best_dir_from_sweep(path)
    mean_batch = read_single_col(Path(best_dir) / f'batch_{metric_name}.txt')
    mean_test = read_single_col(Path(best_dir) / f'test_epoch_{metric_name}.txt')
    std_batch, std_test = np.zeros_like(mean_batch), np.zeros_like(mean_test)

  else:
    mean_batch = read_single_col(path / f'batch_{metric_name}.txt')
    mean_test = read_single_col(path / f'test_epoch_{metric_name}.txt')
    std_batch, std_test = np.zeros_like(mean_batch), np.zeros_like(mean_test)

  # (T,) each
  print(f'read_rundir: {mean_batch.shape=}, {std_batch.shape=}, '
        f'{mean_test.shape=}, {std_test.shape=}')
  return mean_batch, std_batch, mean_test, std_test


def smooth(arr, window_len=101):
  return savgol_filter(arr, window_length=window_len, polyorder=3)


def save_fig(save_fname):
  if save_fname:
    plt.savefig(f'figures/{save_fname}.pdf', bbox_inches='tight')
    plt.savefig(f'figures/{save_fname}.png', bbox_inches='tight', dpi=300)
    print(f'Plots saved to figures/{save_fname}.png and figures/{save_fname}.pdf')


def plot_with_errorbar(x,
                       y,
                       yerr,
                       fmt=None,
                       color=None,
                       alpha_fill=0.15,
                       ax=None,
                       label=None,
                       **line_kwargs):
  ax = ax if ax is not None else plt.gca()
  x, y, yerr = map(np.array, [x, y, yerr])

  if np.isscalar(yerr) or len(yerr) == len(y):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr

  # args and keyword args for plot
  opts, kwopts = [], {}
  if fmt is not None:
    opts.append(fmt)
  if color is not None:
    kwopts['color'] = color

  ax.plot(x, y, *opts, **kwopts, **line_kwargs, label=label)
  ax.fill_between(x, ymax, ymin, **kwopts, alpha=alpha_fill, linewidth=0)


def get_color(alg):
  alg = alg.lower()
  # color_map = {
  #     'SGD': "#f4b273",
  #     'DP-SGD': "#f4b273",
  #     'Adam': '#e24848',
  #     'DP-Adam': '#e24848',
  #     'Adagrad': "#0033cc",
  #     'DP-Adagrad': "#0033cc"
  # }
  color_map = {
      'SGD'.lower() : '#F3938D',
      'DP-SGD'.lower() : '#F3938D',
      'DP_SGD'.lower() : '#F3938D',
      #F3938D, #d62728

      # adagrad tab:purple  #73B3F7
      'Adagrad'.lower() : 'tab:purple',
      'DP-Adagrad'.lower() : 'tab:purple',
      'DP_Adagrad'.lower() : 'tab:purple',

      'RMSProp'.lower() : '#F7B773',
      'DP-RMSProp'.lower() : '#F7B773',
      'DP_RMSProp'.lower() : '#F7B773',

      'variant1'.lower() : '#C738A6',
      'variant2'.lower() : 'tab:green',

      'AdaDPS'.lower(): '#0327C0',
      'PDA-DPMD'.lower(): 'tab:red',
      'KRRT21'.lower(): 'tab:brown',

      'delay2'.lower() : '#17becf',
      'delay2_adagrad'.lower() : '#CF2817',
  }
  algs = list(color_map.keys())
  assert alg in algs, f'{alg=} must one of {algs} (case insensitive)'
  return color_map[alg]


def read_best_dir_from_sweep(sweep_root_dir):
  # NOTE: Does not need to specify min or max, since it's stored in `best_result.txt`
  all_tags = ['lra', 'lrb', 'clipa', 'clipb', 'eps', 'int']
  all_names = ['lr', 'lr2', 'clip1', 'clip2', 'epsilon', 'interval']
  name2tag = dict(zip(all_names, all_tags))

  root_dir = Path(sweep_root_dir)
  with open(root_dir / f'hparams_swept.txt') as f:
    hparams_swept = eval(f.read())  # tuple of strs
  with open(root_dir / f'best_result.txt') as f:
    best_result_dic = eval(f.read())  # dict from params to results

  best_result_hparams = list(best_result_dic.items())[0][0]  # dic has 1 item
  best_result_vals = list(best_result_dic.items())[0][1]
  hparams_tags = [name2tag[name] for name in hparams_swept]
  best_dirname = '_'.join([f'{tag}{val}' for tag, val in zip(hparams_tags, best_result_hparams)])
  best_dir = root_dir / best_dirname
  print(f'Sweep root_dir: {root_dir}\n\tbest dir: {best_dir}\n\tbest_result_dic: {best_result_dic}')
  return best_dir, best_result_vals
