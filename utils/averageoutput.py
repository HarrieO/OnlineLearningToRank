# -*- coding: utf-8 -*-

import numpy as np
import os
import traceback
import json


def cumulative(ranking, discount=0.9995):
  return np.cumsum(discount ** np.arange(ranking.shape[0]) * ranking)


def convert_time(time_in_seconds):
  seconds = time_in_seconds % 60
  minutes = time_in_seconds / 60 % 60
  hours = time_in_seconds / 3600
  return '%02d:%02d:%02d' % (hours, minutes, seconds)


def print_array(array):
  return ' '.join([str(x) for x in array] + ['\n'])


def create_folders(filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

class OutputAverager(object):

  def __init__(self, simulation_arguments):
    self.average_folder = simulation_arguments.average_folder
    self._average_index = 0

  def click_model_name(self, full_name):
    return str(full_name[:full_name.rfind('_')])

  def average_results(self, output_path):
    with open(output_path, 'r') as f:
      sim_args = json.loads(f.readline())
      first_run = json.loads(f.readline())
      run_details = first_run['run_details']

      cur_click_model = self.click_model_name(
        run_details['click model'])
      runtimes = {
          cur_click_model: [float(run_details['runtime'])],
        }

      all_ind = {}
      first_val = {}
      for event in first_run['run_results']:
        iteration = event['iteration']
        for name, val in event.items():
          if name == 'iteration':
            continue
          if name not in all_ind:
            all_ind[name] = []
            first_val[name] = []
          all_ind[name].append(iteration)
          first_val[name].append(val)

      all_val = {}
      for name in all_ind:
        all_ind[name] = np.array(all_ind[name],
                                 dtype=np.int32)
        all_val[name] = {
            cur_click_model: [np.array(first_val[name],
                                       dtype=float)]
          }

      for line in f:
        events = json.loads(line)

        run_details = events['run_details']
        cur_click_model = self.click_model_name(
          run_details['click model'])
        if cur_click_model not in runtimes:
          runtimes[cur_click_model] = []

        runtimes[cur_click_model].append(
          float(run_details['runtime']))

        cur_i = {}
        cur_val = {}
        for name, val in all_ind.items():
          cur_i[name] = 0
          cur_val[name] = np.zeros(val.shape)
          if cur_click_model not in all_val[name]:
            all_val[name][cur_click_model] = []
          all_val[name][cur_click_model].append(cur_val[name])

        for event in events['run_results']:
          iteration = event['iteration']
          for name, val in event.items():
            if name != 'iteration':
              c_i = cur_i[name]
              assert all_ind[name][c_i] == iteration
              cur_val[name][c_i] = val
              cur_i[name] += 1

        for name, val in all_ind.items():
          if name != 'iteration':
            assert cur_i[name] == val.shape[0]

    average_runtimes = {}
    for click_model, values in runtimes.items():
      average_runtimes[click_model] = np.mean(values).tolist()

    results = {}
    for name, cur_ind in all_ind.items():
      cur_results = {
          'indices': cur_ind.tolist()
        }
      results[name] = cur_results
      for click_model, lists in all_val[name].items():
        stacked = np.stack(lists)
        cm_mean = np.mean(stacked, axis=0)
        cm_std = np.std(stacked, axis=0)
        cur_results[click_model] = {
            'mean': cm_mean.tolist(),
            'std': cm_std.tolist(),          
          }

    output = {
      'simulation_arguments': sim_args,
      'runtimes': average_runtimes,
      'results': results
    }

    return output

  def create_average_file(self, sim_output):
    print "opening %s" % sim_output.output_path
    output = self.average_results(sim_output.output_path)

    self.dataset_path = '%s/%s' % (self.average_folder, sim_output.dataset_name)
    self.output_path = '%s/%s.out' % (self.dataset_path, sim_output.simulation_name)
    create_folders(self.dataset_path)
    create_folders(self.output_path)
    with open(self.output_path, 'w') as w:
      w.write(json.dumps(output))
      print 'Closed %d: %s on %s was averaged and stored.' % (self._average_index,
          sim_output.simulation_name, sim_output.dataset_name)

    self._average_index += 1

class IndependentOutputAverager(OutputAverager):
  def __init__(self, average_folder):
    self.average_folder = average_folder
    self._average_index = 0
