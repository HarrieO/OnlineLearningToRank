# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from algorithms.DBGD.pdbgd import P_DBGD
from models.linearmodel import LinearModel


# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_MGD(P_DBGD):

  def __init__(self, n_candidates, *args, **kargs):
    super(P_MGD, self).__init__(*args, **kargs)
    self.n_candidates = n_candidates
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = self.learning_rate,
                             n_candidates = self.n_candidates)


  @staticmethod
  def default_parameters():
    parent_parameters = P_DBGD.default_parameters()
    parent_parameters.update({
      'n_candidates': 49,
      })
    return parent_parameters
