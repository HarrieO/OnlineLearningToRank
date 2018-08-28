# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.neuralmodel import NeuralModel
from algorithms.PDGD.pdgd import PDGD

# Pairwise Differentiable Gradient Descent
class DeepPDGD(PDGD):

  def __init__(self, hidden_layers, *args, **kargs):
    super(DeepPDGD, self).__init__(*args, **kargs)
    self.model = NeuralModel(n_features = self.n_features,
                             learning_rate = self.learning_rate,
                             learning_rate_decay = self.learning_rate_decay,
                             hidden_layers = hidden_layers)

  @staticmethod
  def default_parameters():
    parent_parameters = PDGD.default_parameters()
    parent_parameters.update({
      'learning_rate': 0.01,
      'hidden_layers': [64],
      })
    return parent_parameters
