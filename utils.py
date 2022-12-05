import os
import time


def print_log(message, fpath=None, stdout=False, print_time=False):
  if print_time:
    timestr = time.strftime('%Y-%m-%d %a %H:%M:%S')
    message = f'{timestr} | {message}'
  if stdout:
    print(message)
  if fpath is not None:
    with open(fpath, 'a') as f:
      print(message, file=f)
