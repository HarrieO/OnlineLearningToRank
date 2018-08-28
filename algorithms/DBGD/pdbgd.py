# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import utils.rankings as rnk
from algorithms.DBGD.tddbgd import TD_DBGD
from multileaving.ProbabilisticMultileave import ProbabilisticMultileave

# Probabilistic Interleaving Dueling Bandit Gradient Descent
class P_DBGD(TD_DBGD):

  def __init__(self, PM_n_samples, PM_tau, *args, **kargs):
    super(P_DBGD, self).__init__(*args, **kargs)
    self.multileaving = ProbabilisticMultileave(
                             n_samples = PM_n_samples,
                             tau = PM_tau,
                             n_results=self.n_results)

  @staticmethod
  def default_parameters():
    parent_parameters = TD_DBGD.default_parameters()
    parent_parameters.update({
      'learning_rate': 0.01,
      'learning_rate_decay': 1.0,
      'PM_n_samples': 10000,
      'PM_tau': 3.0,
      })
    return parent_parameters

  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted==False
    self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    inverted_rankings = rnk.rank_single_query(scores,
                                              inverted=True,
                                              n_results=None)
    multileaved_list = self.multileaving.make_multileaving(inverted_rankings)
    return multileaved_list
