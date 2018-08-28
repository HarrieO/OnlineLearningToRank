# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from algorithms.PDGD.pdgd import PDGD

# Pairwise Baseline from Hofmann
class Pairwise(PDGD):

  def __init__(self, epsilon,
               *args, **kargs):
    super(Pairwise, self).__init__(*args, **kargs)
    self.epsilon = epsilon

  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted == False
    n_docs = query_feat.shape[0]
    k = np.minimum(self.n_results, n_docs)
    self.doc_scores = self.model.score(query_feat)

    exploit = rnk.rank_query(self.doc_scores, inverted=False, n_results=k)
    explore = np.random.permutation(np.arange(n_docs))
    coinflips = np.random.uniform(size=k) > self.epsilon

    self.ranking = -np.ones(k, dtype=np.int32)
    exploit_i = 0
    explore_i = 0
    for i in range(k):
      if coinflips[i]:
        while exploit[exploit_i] in self.ranking:
          exploit_i += 1
        self.ranking[i] = exploit[exploit_i]
        exploit_i += 1
      else:
        while explore[explore_i] in self.ranking:
          explore_i += 1
        self.ranking[i] = explore[explore_i]
        explore_i += 1

    self._last_query_feat = query_feat
    return self.ranking

  def _update_to_clicks(self, clicks):
    n_docs = self.ranking.shape[0]
    cur_k = np.minimum(n_docs, self.n_results)

    included = np.ones(cur_k, dtype=np.int32)
    if not clicks[-1]:
      included[1:] = np.cumsum(clicks[::-1])[:0:-1]
    neg_ind = np.where(np.logical_xor(clicks, included))[0]
    pos_ind = np.where(clicks)[0]

    n_pos = pos_ind.shape[0]
    n_neg = neg_ind.shape[0]
    n_pairs = n_pos*n_neg

    if n_pairs == 0:
      return

    pos_r_ind = self.ranking[pos_ind]
    neg_r_ind = self.ranking[neg_ind]

    all_w = np.zeros(n_pos + n_neg)
    all_w[:n_pos] = n_neg
    all_w[n_pos:] = -n_pos

    all_ind = np.concatenate([pos_r_ind, neg_r_ind])

    self.model.update_to_documents(all_ind,
                                   all_w)