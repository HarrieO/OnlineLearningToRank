# -*- coding: utf-8 -*-

import numpy as np


class PairwisePreferenceMultileave(object):

    def __init__(self, num_data_features, k=10):
        self._name = 'Pairwise Preferences Multileave'
        self._k = k
        self.needs_inverted = True
        self.needs_descending = True
        self.needs_oracle = False
        self.vector_aggregation = False

    def clean(self):
        del self._last_inverted_rankings

    def top_rank(self, multileaving, top_docs):
        n_disp = multileaving.shape[0]
        top_rank = np.zeros(n_disp, dtype=np.int32)
        top_rank[:] = n_disp
        for i in range(n_disp):
            in_rank = np.in1d(multileaving, top_docs[:,i])
            top_rank[in_rank] = np.minimum(top_rank[in_rank],i)
        return top_rank

    def make_multileaving(self, descending_rankings, inverted_rankings):
        self._last_inverted_rankings = inverted_rankings
        self._last_descending_rankings = descending_rankings
        self._last_n_rankers = inverted_rankings.shape[0]

        n_docs = descending_rankings.shape[1]
        n_rankers = descending_rankings.shape[0]
        length = min(self._k,n_docs)
        multileaving = np.zeros(length, dtype=np.int32)
        previous_set = np.array([], dtype=np.int32)
        previous_results = {}
        self._last_choice_sizes = np.zeros(length)
        for i in range(length):
            full_set = np.unique(descending_rankings[:,:i+1])
            cur_set = np.setdiff1d(full_set, multileaving[:i], assume_unique=True)
            multileaving[i] = np.random.choice(cur_set,1)
            self._last_choice_sizes[i] = cur_set.shape[0]
        self._last_top_ranks = self.top_rank(multileaving, descending_rankings)
        return multileaving

    def infer_preferences(self, result_list, clicked_docs):
        if np.any(clicked_docs):
            return self.preferences_of_list(result_list, clicked_docs.astype(bool))
        else:
            return np.zeros((self._last_n_rankers, self._last_n_rankers))

    def preferences_of_list(self, result_list, clicked_docs):
        n_disp = result_list.shape[0]
        n_rankers = self._last_n_rankers
        included = np.ones(min(self._k, clicked_docs.shape[0]))
        if not clicked_docs[-1]:
            included[1:] = np.cumsum(clicked_docs[::-1])[:0:-1]
        neg_pref = np.where(np.logical_xor(clicked_docs, included))[0]
        pos_pref = np.where(clicked_docs)[0]

        pair_neg = np.repeat(neg_pref, pos_pref.shape[0])
        pair_pos = np.tile(pos_pref, neg_pref.shape[0])

        pair_min_pos  = np.minimum(pair_pos, pair_neg)
        pair_max_rank = np.maximum(self._last_top_ranks[pair_neg], self._last_top_ranks[pair_pos])
        allowed_pairs = pair_min_pos >= pair_max_rank

        n_allowed_pairs = np.sum(allowed_pairs)
        if n_allowed_pairs > 0:
            pos_allow = pair_pos[allowed_pairs]
            neg_allow = pair_neg[allowed_pairs]
            pair_ind_pos = result_list[pos_allow]
            pair_ind_neg = result_list[neg_allow]

            pair_prob_comp = np.zeros(n_allowed_pairs)
            for i in range(n_allowed_pairs):
                pair_top = sorted([self._last_top_ranks[pos_allow[i]],self._last_top_ranks[neg_allow[i]]])
                pair_prob_comp[i] = 1./np.prod(1.-1./self._last_choice_sizes[pair_top[0]:pair_top[1]])

            correct_pairs = self._last_inverted_rankings[:, pair_ind_neg] \
                                - self._last_inverted_rankings[:, pair_ind_pos] > 0

            total_correct = np.sum(correct_pairs * pair_prob_comp, axis=1) \
                              / n_allowed_pairs

        else:
            total_correct = np.zeros(self._last_inverted_rankings.shape[0])

        return total_correct[:,None] - total_correct[None,:]
