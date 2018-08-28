# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.linearmodel import LinearModel
from algorithms.basiconlineranker import BasicOnlineRanker

# Pairwise Differentiable Gradient Descent
class PDGD(BasicOnlineRanker):

  def __init__(self, learning_rate, learning_rate_decay,
               *args, **kargs):
    super(PDGD, self).__init__(*args, **kargs)
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.model = LinearModel(n_features = self.n_features,
                             learning_rate = learning_rate,
                             learning_rate_decay = learning_rate_decay,
                             n_candidates = 1)


  @staticmethod
  def default_parameters():
    parent_parameters = BasicOnlineRanker.default_parameters()
    parent_parameters.update({
      'learning_rate': 0.1,
      'learning_rate_decay': 1.0,
      })
    return parent_parameters

  def get_test_rankings(self, features,
                        query_ranges, inverted=True):
    scores = -self.model.score(features)
    return rnk.rank_multiple_queries(
                      scores,
                      query_ranges,
                      inverted=inverted,
                      n_results=self.n_results)

  def _create_train_ranking(self, query_id, query_feat, inverted):
    assert inverted == False
    n_docs = query_feat.shape[0]
    k = np.minimum(self.n_results, n_docs)
    self.doc_scores = self.model.score(query_feat)
    self.doc_scores += 18 - np.amax(self.doc_scores)
    self.ranking = self._recursive_choice(np.copy(self.doc_scores),
                                          np.array([], dtype=np.int32),
                                          k)
    self._last_query_feat = query_feat
    return self.ranking

  def _recursive_choice(self, scores, incomplete_ranking, k_left):
    n_docs = scores.shape[0]
    scores[incomplete_ranking] = np.amin(scores)
    scores += 18 - np.amax(scores)
    exp_scores = np.exp(scores)
    exp_scores[incomplete_ranking] = 0
    probs = exp_scores/np.sum(exp_scores)
    safe_n = np.sum(probs > 10**(-4)/n_docs)
    safe_k = np.minimum(safe_n, k_left)

    next_ranking = np.random.choice(np.arange(n_docs),
                                    replace=False,
                                    p=probs,
                                    size=safe_k)
    ranking = np.concatenate((incomplete_ranking, next_ranking))

    k_left = k_left - safe_k
    if k_left > 0:
      return self._recursive_choice(scores, ranking, k_left)
    else:
      return ranking

  def update_to_interaction(self, clicks):
    if np.any(clicks):
      self._update_to_clicks(clicks)

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

    pos_scores = self.doc_scores[pos_r_ind]
    neg_scores = self.doc_scores[neg_r_ind]

    log_pair_pos = np.tile(pos_scores, n_neg)
    log_pair_neg = np.repeat(neg_scores, n_pos)

    pair_trans = 18 - np.maximum(log_pair_pos, log_pair_neg)
    exp_pair_pos = np.exp(log_pair_pos + pair_trans)
    exp_pair_neg = np.exp(log_pair_neg + pair_trans)

    pair_denom = (exp_pair_pos + exp_pair_neg)
    pair_w = np.maximum(exp_pair_pos, exp_pair_neg)
    pair_w /= pair_denom
    pair_w /= pair_denom
    pair_w *= np.minimum(exp_pair_pos, exp_pair_neg)

    pair_w *= self._calculate_unbias_weights(pos_ind, neg_ind)

    reshaped = np.reshape(pair_w, (n_neg, n_pos))
    pos_w =  np.sum(reshaped, axis=0)
    neg_w = -np.sum(reshaped, axis=1)

    all_w = np.concatenate([pos_w, neg_w])
    all_ind = np.concatenate([pos_r_ind, neg_r_ind])

    self.model.update_to_documents(all_ind,
                                   all_w)

  def _calculate_unbias_weights(self, pos_ind, neg_ind):
    ranking_prob = self._calculate_observed_prob(pos_ind, neg_ind,
                                                 self.doc_scores)
    flipped_prob = self._calculate_flipped_prob(pos_ind, neg_ind,
                                                self.doc_scores)
    return flipped_prob / (ranking_prob + flipped_prob)

  def _calculate_observed_prob(self, pos_ind, neg_ind, doc_scores):
    n_pos = pos_ind.shape[0]
    n_neg = neg_ind.shape[0]
    n_pairs = n_pos * n_neg
    n_results = self.ranking.shape[0]
    n_docs = doc_scores.shape[0]

    results_i = np.arange(n_results)
    pair_i = np.arange(n_pairs)
    doc_i = np.arange(n_docs)

    pos_pair_i = np.tile(pos_ind, n_neg)
    neg_pair_i = np.repeat(neg_ind, n_pos)

    min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
    max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
    range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                max_pair_i[:, None] >= results_i)

    safe_log = np.tile(doc_scores[None, :],
                       [n_results, 1])

    mask = np.zeros((n_results, n_docs))
    mask[results_i[1:], self.ranking[:-1]] = True
    mask = np.cumsum(mask, axis=0).astype(bool)

    safe_log[mask] = np.amin(safe_log)
    safe_max = np.amax(safe_log, axis=1)
    safe_log -= safe_max[:, None] - 18
    safe_exp = np.exp(safe_log)
    safe_exp[mask] = 0

    ranking_log = doc_scores[self.ranking] - safe_max + 18
    ranking_exp = np.exp(ranking_log)

    safe_denom = np.sum(safe_exp, axis=1)
    ranking_prob = ranking_exp/safe_denom

    tiled_prob = np.tile(ranking_prob[None, :], [n_pairs, 1])

    safe_prob = np.ones((n_pairs, n_results))
    safe_prob[range_mask] = tiled_prob[range_mask]

    safe_pair_prob = np.prod(safe_prob, axis=1)

    return safe_pair_prob

  def _calculate_flipped_prob(self, pos_ind, neg_ind, doc_scores):
    n_pos = pos_ind.shape[0]
    n_neg = neg_ind.shape[0]
    n_pairs = n_pos * n_neg
    n_results = self.ranking.shape[0]
    n_docs = doc_scores.shape[0]

    results_i = np.arange(n_results)
    pair_i = np.arange(n_pairs)
    doc_i = np.arange(n_docs)

    pos_pair_i = np.tile(pos_ind, n_neg)
    neg_pair_i = np.repeat(neg_ind, n_pos)

    flipped_rankings = np.tile(self.ranking[None, :],
                               [n_pairs, 1])
    flipped_rankings[pair_i, pos_pair_i] = self.ranking[neg_pair_i]
    flipped_rankings[pair_i, neg_pair_i] = self.ranking[pos_pair_i]

    min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
    max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
    range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                max_pair_i[:, None] >= results_i)

    flipped_log = doc_scores[flipped_rankings]

    safe_log = np.tile(doc_scores[None, None, :],
                       [n_pairs, n_results, 1])

    results_ij = np.tile(results_i[None, 1:], [n_pairs, 1])
    pair_ij = np.tile(pair_i[:, None], [1, n_results-1])
    mask = np.zeros((n_pairs, n_results, n_docs))
    mask[pair_ij, results_ij, flipped_rankings[:, :-1]] = True
    mask = np.cumsum(mask, axis=1).astype(bool)

    safe_log[mask] = np.amin(safe_log)
    safe_max = np.amax(safe_log, axis=2)
    safe_log -= safe_max[:, :, None] - 18
    flipped_log -= safe_max - 18
    flipped_exp = np.exp(flipped_log)

    safe_exp = np.exp(safe_log)
    safe_exp[mask] = 0
    safe_denom = np.sum(safe_exp, axis=2)
    safe_prob = np.ones((n_pairs, n_results))
    safe_prob[range_mask] = (flipped_exp/safe_denom)[range_mask]

    safe_pair_prob = np.prod(safe_prob, axis=1)

    return safe_pair_prob

