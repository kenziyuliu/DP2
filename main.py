import argparse
import importlib
import os
import pprint
from pathlib import Path
import sys
import time
import multiprocessing as mp

import numpy as np

import utils
import parallel_utils

METHODS = [
    # Non-private
    'sgd',
    'rmsprop',
    'adagrad',
    'adam',
    # Private baselines
    'dp_sgd',
    'dp_rmsprop',
    'dp_adagrad',
    'dp_adam',
    # DP^2
    'dp2_adagrad',
    'dp2_rmsprop',
    # Prev methods & ablations
    'dp_sgd_mirror',  # PDA-DPMD: https://arxiv.org/abs/2112.00193
    'dp_rmsprop_adadps',  # AdaDPS-RMSProp
    'dp_adagrad_grad_projection',  # KRRT'21: http://proceedings.mlr.press/v134/kairouz21a/kairouz21a.pdf
    'dp2_rmsprop_ablation1',  # Ablation variant 1 of DP^2
    'dp2_adagrad_ablation1',
    'dp2_rmsprop_noise_then_precond',  # Ablation variant 2 of DP^2
    'dp2_adagrad_noise_then_precond',
]

DATASETS = ['imdb', 'so_tag', 'movielens']
CLASSIFICATION_DATASETS = ['imdb', 'so_tag']
MAX_METRIC_DATASETS = ['imdb', 'so_tag']


def read_options():

  parser = argparse.ArgumentParser()
  parser.add_argument('--method', help='training method', type=str, choices=METHODS, default='sgd')
  parser.add_argument('--dataset',
                      help='the name of the dataset',
                      type=str,
                      choices=DATASETS,
                      default='imdb')
  parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
  parser.add_argument('--lr2',
                      type=float,
                      help='2nd LR if needed. Check impl for where this is used.')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--test_batch_size', type=int)
  parser.add_argument('--epochs', help='number of epochs', type=int, default=100)
  parser.add_argument('--eval_every_epoch', type=int, default=1)
  parser.add_argument('--train_metrics_every_iter',
                      type=int,
                      default=20,
                      help='Frequency (# iters) for saving train batch metrics')
  parser.add_argument('--train_epoch_eval',
                      action='store_true',
                      help='Evaluate on the training set after every epoch')

  parser.add_argument('--seed', help='root seed', type=int, default=0)
  parser.add_argument('--sigma',
                      help='noise multiplier of gaussian machenism (== noise_std / l2_bound)',
                      type=float,
                      default=1.0)
  parser.add_argument('--delta', help='delta in the privacy parameters', type=float)
  parser.add_argument('--clip1', help='max l2 norm of the gradient norm', type=float, default=0.1)
  parser.add_argument('--clip2',
                      help='max l2 norm of the preconditioned gradients',
                      type=float,
                      default=1)
  parser.add_argument('--beta1', help='momentum parameter (close to 1)', type=float, default=0.9)
  parser.add_argument('--beta2',
                      help='2nd momentum param (close to 1); e.g. for Adam',
                      type=float,
                      default=0.999)
  parser.add_argument('--rmsprop_gamma',
                      help='momentum for RMSProp preconditioner',
                      type=float,
                      default=0.9)
  parser.add_argument('--reg_lambda', help='l2 regularization parameter', type=float, default=0.0)
  parser.add_argument('--epsilon',
                      help='epsilon value in the preconditioner denominator (e.g. adam)',
                      type=float,
                      default=1e-8)
  parser.add_argument('--interval', help='staleness interval', type=int, default=100)
  parser.add_argument('-o', '--outdir', help='output directory', type=str)
  parser.add_argument('-r',
                      '--repeat',
                      help='number of repetitions on the same GPU',
                      type=int,
                      default=1)
  parser.add_argument('--no_bar', help='Disable progress bar (no `tqdm`)', action='store_true')
  parser.add_argument('--ndigits', help='Number of digits for metric values', type=float, default=6)
  parser.add_argument('--gpu_id',
                      help='specify which single gpu to use; overwrites CUDA_VISIBLE_DEVICES',
                      type=int)

  ## Hyperparam sweeps
  parser.add_argument('--sweep', help='Enable hyperparameter grid sweeping', action='store_true')
  parser.add_argument('--max_procs', help='Max number of processes', type=int, default=20)
  parser.add_argument('--gpu_ids',
                      help='specify which gpus to use; overwrites CUDA_VISIBLE_DEVICES',
                      nargs='+',
                      type=int)
  parser.add_argument('--clip1s',
                      help='Sweep the 1st clip bound (see --clip1 for info)',
                      nargs='+',
                      type=float)
  parser.add_argument('--clip2s',
                      help='Sweep the 1st clip bound (see --clip1 for info)',
                      nargs='+',
                      type=float)
  parser.add_argument('--lrs', help='Sweep the 1st LR (see --lr for info)', nargs='+', type=float)
  parser.add_argument('--lr2s', help='Sweep the 2nd LR (see --lr2 for info)', nargs='+', type=float)
  parser.add_argument('--epsilons',
                      help='Sweep the adaptivity for preconditioners (see --epsilon for info)',
                      nargs='+',
                      type=float)
  parser.add_argument('--intervals',
                      help='Sweep the delay interval (see --interval for info)',
                      nargs='+',
                      type=int)

  ## Dataset/task specific flags
  parser.add_argument('--so_tag_cache',
                      action='store_true',
                      help='Speed up so_tag by fitting the entire dataset in memory')
  parser.add_argument('--matfac_dim_1', type=int, default=943)  # movielens dimensions
  parser.add_argument('--matfac_dim_2', type=int, default=1682)
  parser.add_argument('--matfac_embed_dim', type=int, default=100)
  parser.add_argument('--matfac_density', type=float, default=0.05)

  ## Baseline specific flags
  parser.add_argument('--public_data_frac', type=float, default=0.01)
  parser.add_argument('--grad_proj_k', type=int, default=50)

  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  print(f'Command executed: python3 {" ".join(sys.argv)}')

  # Sanitize args
  args.sigma = args.sigma or 0.0
  args.is_classification = args.dataset in CLASSIFICATION_DATASETS
  args.test_batch_size = args.test_batch_size or min(1024, args.batch_size * 8)
  args.lr2 = args.lr2 or args.lr  # Default lr2 to lr1 if not provided.

  args_dict = vars(args)
  maxLen = max([len(ii) for ii in args_dict.keys()])
  fmtString = '\t%' + str(maxLen) + 's : %s'
  print('Input arguments:')
  for keyPair in sorted(args_dict.items()):
    print(fmtString % keyPair)

  # Logging
  if args.outdir is None:
    print(f'Outdir not provided.', end=' ')
    args.outdir = f'logs/scratch/{args.dataset}-{args.method}-{time.strftime("%Y-%m-%d--%H-%M-%S")}'
  os.makedirs(args.outdir, exist_ok=True)
  print(f'Storing outputs to {args.outdir}')

  # Save the command and the args to file and print the args
  with open(Path(args.outdir) / 'args.txt', 'w') as f:
    pprint.pprint(vars(args), stream=f)
  with open(Path(args.outdir) / 'command.txt', 'w') as f:
    print(' '.join(sys.argv), file=f)

  return args_dict


def runner(options, run_idx=None):
  # Use a specific GPU if specified.
  if options['gpu_id'] is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])

  options['is_parallel'] = (run_idx is not None)
  if options['is_parallel']:
    # Parallel workers w/ worker-specific configs
    options['seed'] += 1234 * run_idx
    options['outdir'] = Path(options['outdir']) / f'run{run_idx}'
    os.makedirs(options['outdir'], exist_ok=True)
  options['run_idx'] = run_idx or 0
  print(f'Run {run_idx or 0} uses root seed {options["seed"]}')

  seed = options['seed']
  np.random.seed(seed + 4321)
  trainer_module = importlib.import_module(f'trainers.{options["method"]}')
  trainer_class = getattr(trainer_module, 'Trainer')
  print(f'[INFO] Trainer: {trainer_module}')
  trainer = trainer_class(options)
  return trainer.train()


def main():
  options = read_options()
  # Single vs repeated runs
  num_repeats = options['repeat']
  start_time = time.perf_counter()
  print('Current time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

  # NOTE: Hyperparameter sweeps do NOT do repeated runs.
  # We can do repetitions once we found the best hyperparams from the sweep.

  if options['sweep']:
    # `trajectories`: dict[hparam_tuple -> worker_fn.output]
    hparam_names, trajectories = parallel_utils.sweep_grid(runner, options)

    # Summarize final results by taking last window avg
    def last_avg(x, window):
      if len(x) == 0:
        return 0
      return x[-window:].mean(axis=0).astype(float).round(options['ndigits']).tolist()

    full_results = {}
    for hparam_tuple, (_batch, _train, _test) in trajectories.items():
      batch_final = last_avg(_batch, window=50)
      train_final = last_avg(_train, window=4)
      test_final = last_avg(_test, window=4)
      full_results[hparam_tuple] = (batch_final, train_final, test_final)  # (3, <num_metrics>)
    # Pick best run by train batch metric (x[1][0][-1]) or eval metric (x[1][2][-1]);
    # use the last metric [-1] in the metric tuple (loss, acc, ppl)
    result_selector = lambda x: x[1][0][-1]
    rank_fn = max if options['dataset'] in MAX_METRIC_DATASETS else min
    best_hparam_tuple, best_result = rank_fn(full_results.items(), key=result_selector)
    best_result = {best_hparam_tuple: best_result}

    out_dir = Path(options['outdir'])
    with open(out_dir / 'hparams_swept.txt', 'w') as f:
      pprint.pprint(hparam_names, stream=f)
    with open(out_dir / 'full_result.txt', 'w') as f:
      pprint.pprint(full_results, stream=f)
    with open(out_dir / 'best_result.txt', 'w') as f:
      pprint.pprint(best_result, stream=f)

    # Also save best runs across each hparam separately
    for hparam_idx, hparam_name in enumerate(hparam_names):
      hparam_best = {}
      for hparam_val in options[f'{hparam_name}s']:  # sweep appends 's'
        pairs = [pair for pair in full_results.items() if pair[0][hparam_idx] == hparam_val]
        best_pair = rank_fn(pairs, key=result_selector)
        hparam_best[hparam_val] = best_pair
      with open(out_dir / f'agg_{hparam_name}_result.txt', 'w') as f:
        pprint.pprint(hparam_best, stream=f)

    print(f'Swept hparams: {hparam_names}')
    print(f'Test metric w/ best hparams: {best_result}')

  elif num_repeats > 1:
    parallel_utils.repeat(runner, num_repeats, options)

  else:
    print('Performing a single run...')
    runner(options)

  end_time = time.perf_counter()
  time_in_mins = (end_time - start_time) / 60
  print(f'Finish training ({time_in_mins:.2f} mins), output stored at {options["outdir"]}')
  print('Current time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))


if __name__ == "__main__":
  main()
