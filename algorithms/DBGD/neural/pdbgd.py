# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils.rankings as rnk
from models.evolutionneuralmodel import EvolutionNeuralModel
from algorithms.DBGD.pdbgd import P_DBGD

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class Neural_P_DBGD(P_DBGD):

  def __init__(self, learning_rate, learning_rate_decay,
               hidden_layers, *args, **kargs):
    super(Neural_P_DBGD, self).__init__(learning_rate = learning_rate,
                                        learning_rate_decay = learning_rate_decay,
                                        *args, **kargs)
    self.model = EvolutionNeuralModel(
                             n_features = self.n_features,
                             learning_rate = learning_rate,
                             n_candidates = 1,
                             learning_rate_decay = learning_rate_decay,
                             hidden_layers = hidden_layers)

  @staticmethod
  def default_parameters():
    parent_parameters = P_DBGD.default_parameters()
    parent_parameters.update({
      'learning_rate': 0.01,
      'learning_rate_decay': 1.0,
      'PM_n_samples': 10000,
      'PM_tau': 3.0,
      'hidden_layers': [64],
      })
    return parent_parameters
