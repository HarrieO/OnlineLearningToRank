# -*- coding: utf-8 -*-

import json
import os
import sys
import time
from datetime import timedelta

def create_folders(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

class FileOutput(object):

    def __init__(self, output_file_path, output_header=None, close_between_writes=False,
                 also_print=False, write_date=False):
        self._output_file_path = output_file_path
        self._close_between_writes = close_between_writes
        self._also_print = also_print
        self._original_stdout = sys.stdout
        self.write_date = write_date
        create_folders(self._output_file_path)
        self._output_file = open(self._output_file_path, 'w')
        self._file_open = True
        self._new_line = True
        self._closed = False
        if not output_header is None:
            self.write(output_header)
        self._end_write()

    def _open_file(self):
        if not self._file_open:
            self._output_file = open(self._output_file_path, 'a')
            self._file_open = True

    def _close_file(self):
        self._output_file.close()
        self._file_open = False

    def _end_write(self):
        if self._close_between_writes:
            self._close_file()

    def _write_str_to_file(self, output_str):
        self._output_file.write(output_str)
        self._new_line = output_str[-1] == '\n'

    def flush(self):
        if self._also_print:
            self._original_stdout.flush()
        self._output_file.flush()

    def write(self, output, skip_write_date=False):
        assert not self._closed
        # if isinstance(output, str):
        #     output = [output]
        # elif isinstance(output, list):
        #     output = [line + '\n' for line in output]
        # assert type(output) is list, 'Expected output to be list, found %s' % type(output)
        self._open_file()
        for line in output:
            if self.write_date and self._new_line and not skip_write_date:
                line = '%s: %s' % (time.strftime('%c'), str(line))
            # assert type(line) is str, 'Output element %s is not a str' % line
            self._write_str_to_file(str(line))
            if self._also_print:
                self._original_stdout.write(line)
        self._end_write()

    def close(self):
        self._close_file()
        self._closed = True
        if self._also_print:
            self._original_stdout.write('\n')


class PrintOutput(object):

    def __init__(self, output_header=None, write_date=False):
        self.write_date = write_date
        if not output_header is None:
            self.write(output_header)

    def write(self, output):
        if type(output) is str:
            output = [output]
        assert type(output) is list, 'Expected output to be list, found %s' % type(output)
        for line in output:
            if self.write_date:
                line = '%s: %s' % (time.strftime('%c'), line)
            print line

    def close(self):
        pass


def get_simulation_report(simulation_arguments):
    file_name = sys.argv[0]
    if file_name[-3:] == ".py":
        file_name = file_name[:-3].split("/")[-1]
    date_str = file_name + "-" + time.strftime('Log-%y-%m-%d-%X')

    if not simulation_arguments.log_folder is None \
        and os.path.isdir(simulation_arguments.log_folder):
        output_path = simulation_arguments.log_folder + '/' + date_str.replace(' ', '-') + '.txt'
        header = ['Starting simulation at %s.' % date_str, 'Log is also stored in output file at %s'
                   % output_path]
        return FileOutput(output_path, output_header=header, also_print=True, write_date=True)
    else:
        header = ['Starting simulation.',
                  'WARNING: No log folder found, log is not stored elsewhere.']
        return PrintOutput(output_header=header, write_date=True)


class SimulationOutput(object):

    """
    Class designed to manage the multiprocessing of simulations over multiple datasets.
    """

    def __init__(self, simulation_arguments, simulation_name, dataset, num_click_models,
                 ranker_arguments, output_averager):
        self._start_time = time.time()
        self.run_index = 0
        self.output_folder = simulation_arguments.output_folder
        self.simulation_name = simulation_name
        self.dataset_name = dataset.name
        self.output_averager = output_averager
        self.print_output = simulation_arguments.print_output
        self._expected_runs = dataset.num_runs_per_fold * dataset.num_folds * num_click_models
        self._closed = False
        self.output_path = '%s/%s/%s.out' % (self.output_folder, self.dataset_name,
                                             self.simulation_name)
        combined_args = {
                'simulation_arguments': vars(simulation_arguments),
                'ranker_arguments': ranker_arguments,
            }
        if self.print_output:
            output_header = json.dumps(combined_args, sort_keys=True,
                                       indent=4, separators=(',', ': '))
            self.file_output = BufferPrintOutput(output_header=output_header)
        else:
            output_header = json.dumps(combined_args, separators=(',',':'))
            self.file_output = FileOutput(self.output_path, output_header=output_header,
                                          close_between_writes=True, also_print=False,
                                          write_date=False)

    def expected_runs(self):
        return self._expected_runs

    def finished(self):
        return self._closed and self.run_index == self._expected_runs

    def write_run_output(self, run_output):
        assert not self._closed, 'Simulation Output (%s) written to after being closed.' \
            % self.output_path

        if self.print_output:
            # self.file_output.write(json.dumps(run_output, sort_keys=True,
            #                            indent=4, separators=(',', ': ')))
            self.file_output.pretty_run_write(self.run_index, run_output)
        else:
            self.file_output.write('\n%s' % json.dumps(run_output))
        
        self.run_index += 1
        if self.run_index >= self._expected_runs:
            self.close()

    def close(self, output_file=None):
        # self.file_output.write(['--------END--------'])
        # total_time = time.time() - self._start_time
        # seconds = total_time % 60
        # minutes = total_time / 60 % 60
        # hours = total_time / 3600
        # self.file_output.write(['Total time taken %02d:%02d:%02d' % (hours, minutes, seconds)])
        self.file_output.close()
        self._closed = True
        if not self.print_output:
            self.output_averager.create_average_file(self)


class BufferPrintOutput(object):

    def __init__(self, output_header=None):
        self._closed = False
        self._output_list = []
        if not output_header is None:
            self.write(output_header)

    def flush(self):
        pass

    def write(self, output):
        assert not self._closed
        assert type(output) is str, 'Wrong output format %s' % type(output)
        self._output_list.append(output)

    def pretty_run_write(self, run_index, run_output):
      run_details = run_output['run_details']
      run_lines = [
          "RUN: %d" % run_index,
          "DATAFOLD: %s" % run_details['data folder'],
          "CLICK MODEL: %s" % run_details['click model'],
          "RUN TIME: %s (%.02f seconds)" % (timedelta(seconds=run_details['runtime']),
                                            run_details['runtime'])
        ]
      tag = run_details['held-out data']
      for event in run_output['run_results']:
        str_line = str(event['iteration'])
        if 'display' in event:
          str_line += ' DISPLAY: %0.3f' % event['display']
        if 'heldout' in event:
          str_line += ' %s: %0.3f' % (tag, event['heldout'])
        run_lines.append(str_line)
      for line in run_lines:
        self.write(line)

    def close(self):
        self._closed = True
        print 'Run Output\n' + '\n'.join(self._output_list)
        self._output_list = []
