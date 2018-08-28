# -*- coding: utf-8 -*-

import pylab as plt
import numpy as np
import random
import argparse
import os
import json
import datetime

def create_folders(filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))


description = 'Script for displaying graphs from output files.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--pdf_folder', dest='pdf_folder', type=str, required=False, default=None,
          help='Folder to output pdfs into.')

parser.add_argument('--folder_prefix', dest='folder_prefix', type=str, required=False,
          default=None, help='Prefix for folders of the same dataset.')

parser.add_argument('plot_name', type=str, help='Name to save plots under.')

parser.add_argument('output_files', type=str, help='Output files to be parsed.', nargs='+')

args = parser.parse_args()


def create_folders(filename):
  if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

def process_run_name(name):
  name = name.replace('_', '\\_')
  name = name.replace('DeepP-DBGD', 'DBGD (neural)')
  name = name.replace('P-DBGD', 'DBGD')
  name = name.replace('P-MGD', 'MGD')
  name = name.replace('DeepPDGD', 'PDGD (neural)')
  return name


pdf_folder = args.pdf_folder
prefix_plot_name = args.plot_name

folder_structure = {}
if args.folder_prefix:
  for output_file in args.output_files:
    prefix = args.folder_prefix
    assert prefix in output_file
    average_file_name = output_file[output_file.find(prefix) + len(prefix):]
    while average_file_name[0] == '/':
      average_file_name = average_file_name[1:]
    data_folder = average_file_name[:average_file_name.find('/')]
    if data_folder not in folder_structure:
      folder_structure[data_folder] = []
    folder_structure[data_folder].append(output_file)
else:
  folder_structure[None] = args.output_files

to_plot = [
       ('offline', 'heldout'),
      ]

for data_folder in sorted(folder_structure.keys()):
  output_files = folder_structure[data_folder]
  data = {}
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
    assert file_name not in data
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
  print 'start plotting'

  # params = {
  #     'text.latex.preamble': r"\usepackage{lmodern}",
  #     'text.usetex': True,
  #     'font.size': 26,
  #     'font.family': 'lmodern',
  #     'text.latex.unicode': True,
  #     }
  # plt.rcParams.update(params)

  colours = [
    'black',
    'r',
    'b',
    'g',
    'y',
    'c',
    'orange',
    'purple',
    'pink',
    'gray',
    ] * 30

  for plot_name, v_name in to_plot:
    for click_model in click_models:
      fig = plt.figure(figsize=(10.5, 6), linewidth=0.1)
      # fig = plt.figure(figsize=(10.5, 4), linewidth=0.1)
      plt.ioff()
      plt.ylabel('NDCG')
      plt.xlabel('impressions')
      plt.gca().yaxis.set_ticks_position('both')

      labels = []
      max_ind = np.NINF
      for i, file_name in enumerate(file_names):
        file_dict = data[file_name]
        colour = colours[i]

        if v_name not in file_dict:
          if v_name == 'heldout' and 'held-out' in file_dict:
            v_name = 'held-out'
          elif v_name == 'held-out' and 'heldout' in file_dict:
            v_name = 'heldout'
          else:
            print 'not found', v_name, file_dict.keys()
            continue
        v_dict = file_dict[v_name]
        ind = np.array(v_dict['indices'])
        if click_model not in v_dict:
          print 'not found', click_model, v_dict.keys()
          continue
        c_dict = v_dict[click_model]

        max_ind = max(max_ind, np.max(ind))
        mean = np.array(c_dict['mean'])
        std = np.array(c_dict['std'])

        plt.fill_between(ind, mean-std, mean+std, color=colour, alpha=0.2)
        plt.plot(ind, mean, color=colour)
        labels.append(process_run_name(file_name))

      if len(labels) > 0:
        # if v_ind == "TEST INDICES":
          # plt.ylim(.6,.8)
        plt.ylim(.2,0.5)
        plt.xlim(-5, 30000)
        plt.xlim(-500, 1000000)
        # plt.xlim(-5, max_ind)
        # plt.xlim(-5, 100000)
        plt.annotate(click_model, xy=(0.02, 0.90), xycoords='axes fraction')
        if click_model == 'perfect':
          plt.legend(labels, loc=4, fontsize=16, frameon=False, ncol=1)
        # plt.legend(labels, loc=0, fontsize=26, frameon=False, ncol=1)

        if not pdf_folder:
          plt.show()
        else:
          plot_file_name = '%s_%s_%s.pdf' % (prefix_plot_name, plot_name, click_model)
          if not data_folder is None:
            plot_file_name = os.path.join(data_folder, plot_file_name)
            create_folders(os.path.join(pdf_folder, plot_file_name))
          plt.savefig(os.path.join(pdf_folder, plot_file_name), bbox_inches='tight')
          print 'saved', plot_file_name
      plt.close(fig)
    print
