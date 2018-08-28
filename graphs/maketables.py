# -*- coding: utf-8 -*-

import pylab as plt
import numpy as np
import random
import argparse
import os
import json
import datetime

description = 'Script for displaying graphs from output files.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--table_folder', dest='table_folder', type=str, required=False, default=None,
          help='Folder to output pdfs into.')

parser.add_argument('--baselines', dest='baselines', type=str, required=False, default=None,
          help='Folder to output pdfs into.', nargs='+')

parser.add_argument('--folder_prefix', dest='folder_prefix', type=str, required=False,
          default=None, help='Prefix for folders of the same dataset.')

parser.add_argument('plot_name', type=str, help='Name to save plots under.')

parser.add_argument('output_files', type=str, help='Output files to be parsed.', nargs='+')

args = parser.parse_args()

def create_folders(filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

def get_significance(mean_1, mean_2, std_1, std_2, n):
    significance = ''
    ste_1 = std_1 / np.sqrt(n)
    ste_2 = std_2 / np.sqrt(n)
    t = (mean_1 - mean_2) / np.sqrt(ste_1 ** 2 + ste_2 ** 2)
    # treatment is worse than baseline
    # values used are for 120 degrees of freedom
    # (http://changingminds.org/explanations/research/analysis/
    # t-test_table.htm)
    significance = '\\hphantom{\\tiny \\dubbelneer}'
    if mean_1 > mean_2:
        if abs(t) >= 2.62:
            significance = '{\\tiny \\dubbelneer}'
        elif abs(t) >= 1.98:
            significance = '{\\tiny \\enkelneer}'
    else:
        if abs(t) >= 2.62:
            significance = '{\\tiny \\dubbelop}'
        elif abs(t) >= 1.98:
            significance = '{\\tiny \\enkelop}'
    return significance

class OutputTable(object):

    def __init__(self, table_name, table_folder):
        self._closed = False
        self.output_path = '%s/%s.tex' % (table_folder, table_name)
        print 'creating file at %s' % self.output_path
        create_folders(self.output_path)
        self._output_file = open(self.output_path, 'w')
        self.writeline('% !TEX root = ../main.tex')

    def writeline(self, *line):
        full_line = ' '.join(line)
        self._output_file.write(full_line + '\n')
        print full_line

    def write(self, *line):
        full_line = ' '.join(line)
        self._output_file.write(full_line + ' ')
        print full_line,

    def close(self):
        self._closed = True
        self._output_file.close()
        print 'Finished writing to and closed:', self.output_path

def process_run_name(name):
  name = name.replace('_', '\\_')
  name = name.replace('DeepP-DBGD', 'DBGD (neural)')
  name = name.replace('P-DBGD', 'DBGD (linear)')
  name = name.replace('P-MGD', 'MGD (linear)')
  name = name.replace('PDGD', 'PDGD (linear)')
  name = name.replace('DeepPDGD (linear)', 'PDGD (neural)')
  name = name.replace('Pairwise', 'Pairwise (linear)')
  return name

def process_folder_name(name):
  name = name.replace('_', '\\_')
  name = name.replace('Webscope\\_C14\\_Set1', 'Yahoo')

  return name

prefix_plot_name = args.plot_name
folder_structure = {}
if args.folder_prefix:
  for output_file in args.output_files + args.baselines:
    prefix = args.folder_prefix
    assert prefix in output_file
    average_file_name = output_file[output_file.find(prefix) + len(prefix):]
    while average_file_name[0] == '/':
      average_file_name = average_file_name[1:]
    data_folder = average_file_name[:average_file_name.find('/')]
    data_folder = process_folder_name(data_folder)
    if data_folder not in folder_structure:
      folder_structure[data_folder] = []
    folder_structure[data_folder].append(output_file)
else:
  folder_structure[None] = args.output_files

to_table = [
       # ('offline', 'heldout', 10000),
       ('online', 'cumulative-display', 10000),
      ]

baselines = []
methods = []

all_data = {}
for data_folder in sorted(folder_structure.keys()):
  output_files = folder_structure[data_folder]
  data = {}
  all_data[data_folder] = data
  file_names = []
  click_models = []
  value_names = []
  if data_folder is None:
    print 'No data folders found, outputting directly.'
  else:
    print 'Found data folder: %s' % data_folder
  for output_file in output_files:
    print 'reading', output_file
    file_name = output_file.split('/')[-1]
    if file_name[-4:] == '.out':
      file_name = file_name[:-4]
    file_name = process_run_name(file_name)
    if output_file in args.baselines and file_name not in baselines:
      baselines.append(file_name)
    elif output_file not in args.baselines and file_name not in methods:
      methods.append(file_name)
    assert file_name not in data, '%s already in %s' % (file_name, data.keys())
    data[file_name] = {}
    file_names.append(file_name)
    with open(output_file) as f:
      output = json.load(f)
      for name, value in output['runtimes'].items():
        print name,
        print datetime.timedelta(seconds=value),
        print '(%d seconds)' % value
      data[file_name] = output['results']
      for v_name in output['results']:
        if v_name not in value_names:
          value_names.append(v_name)
        for c_m in output['results'][v_name]:
          if c_m == 'indices':
            continue
          if c_m not in click_models:
            click_models.append(c_m)

    print

  print 'finished reading, found the following value types:'
  for name in value_names:
    print name
  print

click_models = ['perfect', 'navigational', 'informational']

folder_order = sorted(folder_structure.keys())
for table_name, table_value, table_ind in to_table:
  table_data = {}
  for folder_name in folder_order:
    all_f_data = all_data[folder_name]
    f_data = {}
    table_data[folder_name] = f_data

    for c_m in click_models:
      c_data = {}
      max_v = np.NINF
      f_data[c_m] = c_data
      for b_name in baselines:
        b_data = all_data[folder_name][b_name][table_value]
        b_ind = np.array(b_data['indices'])
        if np.any(b_ind == table_ind):
          v_i = np.where(b_ind == table_ind)[0][0]
        else:
          diff = b_ind - table_ind
          v_i = np.argmax(diff[diff<=0])
        v_mean = b_data[c_m]['mean'][v_i]
        v_std = b_data[c_m]['std'][v_i]

        max_v = max(max_v, v_mean)
        c_data[b_name] = (v_mean, v_std, None)

      for m_name in methods:
        m_data = all_data[folder_name][m_name][table_value]
        m_ind = np.array(m_data['indices'])
        if np.any(m_ind == table_ind):
          v_i = np.where(m_ind == table_ind)[0][0]
        else:
          diff = b=m_ind - table_ind
          v_i = np.argmax(diff[diff<=0])
        v_mean = m_data[c_m]['mean'][v_i]
        v_std = m_data[c_m]['std'][v_i]

        sig = []
        for b_name in baselines:
          b_mean, b_std, _ = c_data[b_name]
          sig.append(get_significance(b_mean, v_mean, b_std, v_std, 125))

        max_v = max(max_v, v_mean)
        c_data[m_name] = (v_mean, v_std, sig)

      c_data['maximum'] = max_v

  out = OutputTable(table_name, args.table_folder)
  out.writeline('\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill} } l ', 'l '
                  * len(folder_order), '}')
  out.writeline('\\toprule')

  for data_folder in folder_order:
    out.write(' & { \\small \\textbf{%s}}' % data_folder)
  out.writeline('\\\\')

  for click_model in click_models:
    out.writeline('\\midrule')
    out.writeline('& \\multicolumn{%d}{|c|}{\\textit{%s}} \\\\' % (len(folder_order), click_model))
    out.writeline('\\midrule')

    for name in baselines + methods:
      out.write(name)

      for folder in folder_order:
        v_max = round(table_data[folder][click_model]['maximum'], 1)
        v_mean, v_std, v_sig = table_data[folder][click_model][name]
        out.write('&')

        if round(v_mean, 1) >= v_max:
          out.write('\\bf')

        out.write('%0.01f {\\tiny (%0.01f)}' % (v_mean, v_std))
        if not (v_sig is None):
          out.write(' '.join(v_sig))

      out.writeline('\\\\')




  out.writeline('\\bottomrule')
  out.writeline('\\end{tabular*}')
  out.close()

  print
  print
  print




