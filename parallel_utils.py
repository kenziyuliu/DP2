import os
import collections
import itertools
import math
import multiprocessing as mp
from pathlib import Path
import pprint

import numpy as np

import utils


def repeat(worker_fn, num_repeats: int, args: dict):
  assert not args['sweep'], f'Sweeps with repeated runs not supported.'
  print(f'Performing {num_repeats} repeated runs in parallel...')
  out_dir = Path(args['outdir'])
  # GPU: will default to whatever is passed from `gpu_id`

  with mp.Pool(num_repeats + 1) as pool:
    # results: (n_repeat, batch/train_epoch/test_epoch, T, <num_metrics>)
    results = [pool.apply_async(worker_fn, (args, run_idx)) for run_idx in range(num_repeats)]
    results = [r.get() for r in results]

    def save_metrics(metrics, metric_name, last_avg=1):
      """Saves the per-round metrics (either batch / train_epoch / test_epoch)."""
      if metric_name == 'train_metrics' and not args['train_epoch_eval']:
        return

      metrics = np.array(metrics).astype(float)  # (num_repeats, T, <num_metrics>)
      _mean, _std, _med = [agg_fn(metrics, axis=0) for agg_fn in [np.mean, np.std, np.median]]

      # Save aggregate to file; shape (T, <num_metrics>) each
      for tag, agg in zip(['mean', 'std', 'median'], [_mean, _std, _med]):
        utils.print_log(pprint.pformat(agg.astype(float).round(6).tolist()),
                        fpath=out_dir / f'{tag}_{metric_name}.txt')

      summary_fn = lambda x: x[-last_avg:].mean(axis=0).round(6)
      _mean, _std, _med = map(summary_fn, [_mean, _std, _med])
      print(f'{metric_name} mean ± std (median) (last {last_avg} avg):')
      print(f'\tmetrics: {_mean} ± {_std}, ({_med})')

      # Save summary metrics to file; shape (<num_metrics>,) each
      with open(out_dir / f'summary_final_{metric_name}.txt', 'w') as f:
        pprint.pprint(dict(mean=_mean, std=_std, med=_med), stream=f)

    rep_batch, rep_train, rep_test = zip(*results)
    save_metrics(rep_batch, 'batch_metrics', last_avg=50)
    # Since there are repeats, we simply look at the final epoch
    save_metrics(rep_train, 'train_metrics', last_avg=1)
    save_metrics(rep_test, 'test_metrics', last_avg=1)


def sweep_grid(worker_fn, args: dict):
  """Performs a hyperparameter sweep over a grid."""
  out_dir = args['outdir']
  gpu_ids = args['gpu_ids']
  assert args['repeat'] == 1, f'Only sweeps with single runs are supported.'
  # If no GPUs, default to CPU training and feed gpu_id=None to workers.
  if not gpu_ids:
    gpu_ids = [None]

  # NOTE: the folloiwng arg names must be consistent with `argparse`
  all_tags = ['lra', 'lrb', 'clipa', 'clipb', 'eps', 'int']
  all_names = ['lr', 'lr2', 'clip1', 'clip2', 'epsilon', 'interval']
  all_list_names = [name + 's' for name in all_names]
  assert len(all_tags) == len(all_names) == len(all_list_names)

  # Sweep only the specified hparams (proceed even with only one value inside)
  sweep_tuples = []
  for tag, name, list_name in zip(all_tags, all_names, all_list_names):
    if args[list_name] is not None:
      sweep_tuples.append((tag, name, args[list_name]))  # (str, str, list)
  assert len(sweep_tuples) > 0, f'Specified `--sweep` but no hparam lists provided.'

  sweep_tags, sweep_names, sweep_values = zip(*sweep_tuples)
  print(f'[INFO] sweep_tags={sweep_tags}, sweep_names={sweep_names}, sweep_values={sweep_values}')

  # Construct hparam grid with cartesian product
  grid = list(itertools.product(*sweep_values))  # `swee_values` is list[list[num]]
  grid_str = ','.join([f'{tag}={vals}' for tag, vals in zip(sweep_tags, sweep_values)])

  results = collections.defaultdict(list)
  pool_size = args['max_procs']
  num_runs = len(grid)
  num_pools = math.ceil(num_runs / pool_size)
  print(f'[INFO] Grid size {num_runs} w/ ~= {num_pools} pools of size {pool_size}:\n\t{grid_str}')

  # Start sweeping!
  def grid_runner(pool, gpu_id, hparam_tuple):
    # Create "tag1val1_tag2val2_..." string tags
    run_dir = '_'.join([f'{tag}{val}' for tag, val in zip(sweep_tags, hparam_tuple)])
    run_dir = f'{out_dir}/{run_dir}'
    os.makedirs(run_dir, exist_ok=True)
    # For update the sweep hparams
    run_args = {
        **args,
        **dict(zip(sweep_names, hparam_tuple)), 'outdir': run_dir,
        'gpu_id': gpu_id,
        'no_bar': True
    }
    return pool.apply_async(worker_fn, (run_args, ))

  # Reset pool in chunks to prevent memory overload
  grid_pointer, pool_idx = 0, 0
  gpu_id_iter = itertools.cycle(list(gpu_ids))
  while grid_pointer < num_runs:
    chunk = grid[grid_pointer:grid_pointer + pool_size]
    print(f'[INFO] Pool {pool_idx} with {len(chunk)} / {pool_size} procs')
    print(f'[INFO] Running chunks ({sweep_names}):\n{chunk}')

    with mp.Pool(pool_size) as pool:
      # Run in parallel (result is a list of repeats) and collect results
      for hparam_tuple in chunk:
        results[hparam_tuple] = grid_runner(pool, next(gpu_id_iter), hparam_tuple)
      for hparam_tuple in chunk:
        results[hparam_tuple] = results[hparam_tuple].get()

    grid_pointer += len(chunk)
    pool_idx += 1

  print(f'[INFO] Sweep outputs stored to {out_dir}')
  # Results: dict[hparam_tuple -> worker_fn.output]
  return sweep_names, results
