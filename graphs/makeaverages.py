# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.averageoutput import IndependentOutputAverager

def create_folders(filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))


description = 'Script for averaging over full run output files.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--average_folder', dest='average_folder', type=str,
                    required=True, default=None,
                    help='Folder to output pdfs into.')

parser.add_argument('--fullrun_prefix', dest='fullrun_prefix', type=str,
                    required=True, default=None,
                    help='Prefix for folders of full runs of the same dataset.')

parser.add_argument('output_files', type=str, nargs='+',
                    help='Output files to be parsed.')

args = parser.parse_args()


def create_folders(filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

def process_run_name(name):
  name = name.replace('_', '\\_')
  return name


average_folder = args.average_folder
averager = IndependentOutputAverager(average_folder)

path_pairs = []
for output_file in args.output_files:
  prefix = args.fullrun_prefix
  assert prefix in output_file
  average_file_name = output_file[output_file.find(prefix) + len(prefix):]
  while average_file_name[0] == '/':
    average_file_name = average_file_name[1:]
  average_dest = '%s/%s' % (average_folder, average_file_name)
  path_pairs.append((output_file, average_dest))

failed_paths = []
success_paths = []
for source, dest in path_pairs:
  success = True
  try:
    average_results = averager.average_results(source)
  except KeyboardInterrupt:
    raise
  except:
    success = False
    print 'Failed: ', source
    failed_paths.append(source)

  if success:
    print 'Success:', source, '   ->   ', dest

    create_folders(dest)
    with open(dest, 'w') as w:
      w.write(json.dumps(average_results))

    success_paths.append(source)

print
print 'Done processing.'
print
print 'Successfully averaged the following files:'
print
print ' '.join(success_paths)
print
print 'Failed averaging the following files:'
print
print ' '.join(failed_paths)
print
