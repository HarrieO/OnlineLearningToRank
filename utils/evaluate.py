# -*- coding: utf-8 -*-

from numpy import log2
from random import sample
import numpy as np


def get_dcg(ordered_labels):
    return np.sum((2 ** ordered_labels - 1) / np.log2(np.arange(ordered_labels.shape[0]) + 2))


def get_idcg(complete_labels, max_len):
    return get_dcg(np.sort(complete_labels)[:-1 - max_len:-1])


def get_single_ndcg_for_rankers(descending_rankings, document_labels, max_len, idcg=None):
    if idcg == None:
        idcg = get_idcg(document_labels, max_len)
    if idcg == 0:
        return np.zeros(descending_rankings.shape[0])
    return get_single_dcg_for_rankers(descending_rankings, document_labels, max_len)/idcg


def get_single_dcg_for_rankers(descending_rankings, document_labels, max_len):
    displayed_rankings = descending_rankings[:, :max_len]
    displayed_labels = document_labels[displayed_rankings]
    return np.sum((2 ** displayed_labels - 1) / np.log2(np.arange(displayed_labels.shape[1])
                  + 2)[None, :], axis=1)


def evaluate_ranking(ranking, labels, idcg, max_len):
    ordered_labels = labels[ranking]
    if idcg == 0.0:
        return 0.0
    return get_dcg(ordered_labels) / idcg


def evaluate(rankings, label_vector, idcg_vector, n_queries, max_len):
    '''
    Takes rankings as lists of indices, which corresponds to label_lists, lists of label lists.
    '''
    nominators = 2. ** label_vector - 1.
    denominators = np.log2(rankings + 2.)
    nominators[idcg_vector == 0] = 0
    nominators[rankings >= max_len] = 0

    idcg_copy = np.copy(idcg_vector)
    idcg_copy[idcg_vector == 0] = 1

    return np.sum(nominators / denominators / idcg_copy) / n_queries


def get_dcg_from_matrix(label_matrix, n_vector, max_len):
    label_matrix = label_matrix[:, :max_len]

    nominators = 2 ** label_matrix - 1
    nominators[np.arange(max_len)[None, :] >= n_vector[:, None]] = 0

    denominator = np.log2(np.arange(max_len) + 2)
    idcg_vector = np.sum(nominators / denominator[None, :], axis=1)

    return idcg_vector


def get_idcg_list(label_vector, qptr, max_len, spread=False):

    n = qptr[1:] - qptr[:-1]
    max_documents = np.max(n)

    starts = np.zeros(n.shape[0] + 1, dtype=np.int32)
    starts[1:] = np.cumsum(n)

    ind = starts[:-1, None] + np.arange(0, max_documents)[None, :]
    ind = np.minimum(ind, starts[1:, None] - 1)

    label_matrix = label_vector[ind]
    label_matrix[np.arange(max_documents)[None, :] >= n[:, None]] = 0
    label_matrix = np.sort(label_matrix, axis=1)[:, ::-1]

    idcg_list = get_dcg_from_matrix(label_matrix, n, max_len)

    if spread:
        spread_ind = np.zeros(qptr[-1], dtype=np.int32)
        spread_ind[qptr[1:-1]] = 1
        spread_ind = np.cumsum(spread_ind)

        return idcg_list[spread_ind]
    else:
        return idcg_list
