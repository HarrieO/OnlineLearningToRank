# -*- coding: utf-8 -*-

import random
import time
import sharedmem
import datetime
import gc
import sys
import numpy as np
from singlesimulation import SingleSimulation
from multiprocessing import Process, Queue
from Queue import Empty
from utils.clicks import get_click_models
from utils.datasetcollections import get_datasets
from utils.simulationoutput import SimulationOutput, get_simulation_report
from utils.averageoutput import OutputAverager


class DataSimulation(object):

  """
  Class designed to manage the multiprocessing of simulations over multiple datasets.
  """

  def __init__(self, simulation_arguments):
    self.sim_args = simulation_arguments
    self.num_proc = simulation_arguments.n_processing
    self.n_runs = simulation_arguments.n_runs

    self.output_queue = Queue()
    self.single_sims = []
    self.processes = []

    self.folds_in_mem = 0
    self.max_folds = 999

    self.output_averager = OutputAverager(simulation_arguments)
    self.report_output = get_simulation_report(simulation_arguments)
    sys.stdout = self.report_output
    sys.stderr = self.report_output

  def run(self, ranker_tuples):
    starttime = time.time()
    self.active = 0
    self.click_models = {}
    self.run_outputs = {}
    self.all_launched = {}
    self.run_index = 0
    self.read_index = 0
    self.clean_index = 0
    self._launched = 0
    self._outputs_found = 0
    datasets = list(get_datasets(self.sim_args))
    for dataset in datasets:
      self.max_folds = min(self.max_folds, dataset.max_folds)
      if not dataset.click_model_type in self.click_models:
        missing_type = dataset.click_model_type
        missing_models = get_click_models(self.sim_args.click_models
                                          + [dataset.click_model_type])
        self.click_models[missing_type] = missing_models

    for process in self.load_processes(datasets, ranker_tuples):
      self._launched += 1
      process.start()
      while self.update_active() >= self.num_proc:
        self.wait_for_output()

    while self._expecting_output():
      self.wait_for_output()
    self.update_active()

    seconds_past = time.time() - starttime
    print ('Time taken: %s (%d seconds)' %
        (str(datetime.timedelta(seconds=seconds_past)),
         seconds_past))

    assert all(output.finished() for output in self.run_outputs.values()), \
        'Program exiting but not all outputs were finished.'

  def load_processes(self, datasets, ranker_tuples):
    for dataset in datasets:
      for datafold in dataset.get_data_folds(self.sim_args):
        for proc in self.load_datafold_processes(datafold, ranker_tuples):
          yield proc
        self.all_launched[datafold] = True
      while self.folds_in_mem >= dataset.max_folds:
        self.wait_for_output()

  def load_datafold_processes(self, datafold, ranker_tuples):
    while self.folds_in_mem >= datafold.max_folds:
      self.wait_for_output()
      self.update_active()
    print 'Read   %d: Fold %d of dataset %s.' % (self.read_index,
              datafold.fold_num + 1, datafold.name)
    datafold.read_data()
    self.read_index += 1
    self.wait_for_output()
    self.update_active()
    for run_name, r_class, r_new_args in ranker_tuples:
      r_args = r_class.default_parameters()
      r_args.update(r_new_args)
      output_key = run_name, datafold.name
      if not output_key in self.run_outputs:
        self.run_outputs[output_key] = SimulationOutput(
                self.sim_args, run_name, datafold,
                len(self.click_models[datafold.click_model_type]), r_args,
                self.output_averager)
      for c_m in self.click_models[datafold.click_model_type]:
        sim = SingleSimulation(self.sim_args, self.output_queue, c_m, datafold)
        ranker_setup = r_class, r_args
        r_args['n_results'] = self.sim_args.n_results
        r_args['n_features'] = datafold.num_features
        for i in xrange(datafold.num_runs_per_fold):
          new_proc = Process(target=self.start_run, args=(sim, output_key, ranker_setup,
                             self.run_index))
          self.processes.append((new_proc, datafold))
          print 'Launch %d: %s %d with click model %s on fold %d from dataset %s.' % (
              self.run_index,
              run_name,
              i,
              c_m.name,
              datafold.fold_num + 1,
              datafold.name,
              )
          self.run_index += 1
          self.report_output.flush()
          yield new_proc

  def start_run(self, simulation, output_key, ranker_setup, seed=0):
    """
    Performs a single run.
    Random functions get different seeds for each process.
    """
    random.seed((time.time(), seed))
    np.random.seed(int(time.time() + seed * 100 + seed))
    rankerclass, ranker_args = ranker_setup
    ranker = rankerclass(**ranker_args)
    simulation.run(ranker, output_key=output_key)

  def update_active(self):
    """
    Checks how many child processes are still active.
    """
    dead_processes = [p for p in self.processes if not p[0].is_alive()]
    self.processes = [p for p in self.processes if p[0].is_alive()]
    alive_folds = {}
    for _, datafold in self.processes:
      alive_folds[datafold] = True
    self.folds_in_mem = len(alive_folds)

    self.max_folds = min([999] + [datafold.max_folds for datafold in alive_folds])
    self.active = len(self.processes)
    dead_datafolds = {}
    for proc, datafold in dead_processes:
      proc.join()
      if not datafold in alive_folds and datafold in self.all_launched:
        dead_datafolds[datafold] = True

    for datafold in dead_datafolds:
      print 'Clean  %d: Fold %d of dataset %s.' % (self.clean_index, datafold.fold_num + 1,
              datafold.name)
      datafold.clean_data()
      self.clean_index += 1

    # make extra sure that the process is removed from memory
    del dead_processes
    gc.collect()

    # print 'Folds %d max folds %d active %d' % (self.folds_in_mem, self.max_folds, self.active)
    return self.active

  def wait_for_output(self, timeout=30):  # 0):
    """
    Prints output for all finished threads
    """
    found = not self._expecting_output()
    try:
      while True:
        output_key, run_output = self.output_queue.get(block=not found, timeout=timeout)
        found = True
        sim_output = self.run_outputs[output_key]
        print 'Output %d: %s on dataset %s. (%d/%d)' % (self._outputs_found, output_key[0],
                output_key[1], sim_output.run_index+1, sim_output.expected_runs())
        sim_output.write_run_output(run_output)
        self._outputs_found += 1
    except Empty:
        pass
    self.update_active()

  def _expecting_output(self):
    return self._outputs_found < self._launched
