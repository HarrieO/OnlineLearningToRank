# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.linearmodel import LinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
from multileaving.TeamDraftMultileave import TeamDraftMultileave

# Dueling Bandit Gradient Descent
class TD_DBGD(BasicOnlineRanker):

  def __init__(self, learning_rate, learning_rate_decay,
               *args, **kargs):
    super(TD_DBGD, self).__init__(*args, **kargs)
    self.learning_rate = learning_rate
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = learning_rate,
                             n_candidates = 1,
                             learning_rate_decay = learning_rate_decay)
    self.multileaving = TeamDraftMultileave(
                             n_results=self.n_results)


  @staticmethod
  def default_parameters():
    parent_parameters = BasicOnlineRanker.default_parameters()
    parent_parameters.update({
      'learning_rate': 0.01,
      'learning_rate_decay': 1.0,
      })
    return parent_parameters

  def get_test_rankings(self, features,
                        query_ranges, inverted=True):
    scores = self.model.score(features)
    return rnk.rank_multiple_queries(
                      scores,
                      query_ranges,
                      inverted=inverted,
                      n_results=self.n_results)

  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted == False
    self.model.sample_candidates()
    scores = self.model.candidate_score(query_feat)
    rankings = rnk.rank_single_query(scores, inverted=False, n_results=self.n_results)
    multileaved_list = self.multileaving.make_multileaving(rankings)
    return multileaved_list

  def update_to_interaction(self, clicks):
    winners = self.multileaving.winning_rankers(clicks)
    self.model.update_to_mean_winners(winners)
