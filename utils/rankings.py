import numpy as np

def invert_rankings(rankings, dtype=None):
  '''
  Invert indices in a matrix of rankings, ranking per row.
  '''
  if dtype is None:
    inverted = np.zeros(rankings.shape)
  else:
    inverted = np.zeros(rankings.shape, dtype=dtype)
  inverted[np.arange(rankings.shape[0])[:,None],rankings] = np.arange(rankings.shape[1])[None,:]
  return inverted

def invert_ranking(ranking, dtype=None):
  """
  'Inverts' ranking, each element gets the index it has in the ranking.
  [2,0,1] becomes [1,3,0]
  """
  if dtype is None:
    inverted = np.zeros(ranking.shape)
  else:
    inverted = np.zeros(ranking.shape, dtype=dtype)
  inverted[ranking] = np.arange(ranking.shape[0])
  return inverted

def tiebreak_sort(unranked, n_results=None, full_sort=False):
  if full_sort or n_results is None:
    n_results = unranked.shape[-1]
  return _tiebreak_sort(unranked, n_results)

def _tiebreak_sort(unranked, n_results):
  """
  Sorts rows of a matrix using tiebreakers, along the last axis.
  """

  n_axis = len(unranked.shape) 
  assert (n_axis == 1 or n_axis == 2)

  tiebreakers = np.random.random(unranked.shape)
  complex_predictions = np.empty(unranked.shape, dtype=np.complex)
  complex_predictions.real = unranked
  complex_predictions.imag = tiebreakers

  max_n_docs = unranked.shape[-1]
  max_part = np.minimum(n_results, max_n_docs)
  if max_part == max_n_docs:
    return np.argsort(complex_predictions, axis=-1)

  part = np.argpartition(complex_predictions, max_part-1, axis=-1)
  slice_ind = (slice(None),) * (len(unranked.shape)-1)
  slice_ind += (slice(0,max_part),)

  if n_axis == 1:
    part_pred = complex_predictions[part[slice_ind]]
    front_sort = np.argsort(part_pred, axis=-1)
    part[slice_ind] = part[slice_ind][front_sort]
  else:
    extra_ind = np.arange(unranked.shape[0])[:,None]
    part_sliced = part[slice_ind]
    extra_ind = np.empty(part_sliced.shape, dtype=np.int32)
    extra_ind[:,:] = np.arange(unranked.shape[0])[:,None]
    part_pred = complex_predictions[extra_ind, part[slice_ind]]
    front_sort = np.argsort(part_pred, axis=-1)
    part_sliced[:, :] = part_sliced[extra_ind, front_sort]

  return part

def get_score_rankings(weights,feature_matrix,qptr,max_documents=None, inverted=False):
  """
  Given weights and a feature matrix the documents are ranked and scored according to their dot product.
  """
  # minus to reverse ranking
  predictions = -np.squeeze(np.dot(weights.T,feature_matrix))
  return rank_queries(predictions,qptr,max_documents=max_documents,inverted=inverted)

def rank_queries(predictions, qptr, max_documents=None, inverted=False):
  """
  Given predicted scores for queries rankings are generated and returned.
  """

  max_value = np.max(predictions)
  # vector with lenght of each doclist
  n = qptr[1:]-qptr[:-1]
  if not max_documents:
    max_documents = np.max(n)

  # the vector of documents is reshaped into a matrix
  # with a document list on every row
  ind = qptr[:-1,None] + np.arange(0,max_documents)[None,:]
  ind = np.minimum(ind,qptr[1:,None]-1)
  # warped is now a matrix of size n_queries x max_documents
  warped = predictions[ind]
  # every document that appears in a row but not in the query list
  # (due to n_query_list < max_documents) gets the worst score in off all documents
  # this makes sure they do not appear in the final ranking
  warped[np.arange(max_documents)[None,:] >= n[:,None]] = max_value + 1

  # tiebreak sort uses numpy to rank every row in the matrix
  # this is faster than ranking them by seperate calls
  rankings = tiebreak_sort(warped)
  if inverted:
    inverted = invert_rankings(rankings,dtype=np.int32)
    return inverted[np.arange(max_documents)[None,:] < n[:,None]]

  else:
    return rankings[np.arange(max_documents)[None,:] < n[:,None]]

def rank_query(predictions, inverted=False, n_results=None):
  """
  Given predicted scores of a single query returns rankings.
  """
  ranking = tiebreak_sort(predictions, n_results)
  if inverted:
    if len(ranking.shape) == 1:
      return invert_ranking(ranking,dtype=np.int32)
    else:
      return invert_rankings(ranking,dtype=np.int32)
  else:
    return ranking

def rank_candidate_queries(weights,feature_matrix,qptr,n_results=None,inverted=False):
  n_docs = feature_matrix.shape[1]
  scores = -np.dot(weights,feature_matrix)
  qid_per_doc = np.zeros(n_docs, dtype=np.int32)
  qid_per_doc[qptr[1:-1]] = 1
  qid_per_doc = np.cumsum(qid_per_doc)

  index_offset = np.zeros(n_docs, dtype=np.int32)
  index_offset[:] = qptr[qid_per_doc]

  score_offset = (np.max(np.abs(scores),axis=1)+1.)[:,None]*qid_per_doc[None,:]
  scores += score_offset

  descending = rank_query(scores, n_results=n_results)

  if not inverted:
    descending -= index_offset[None,:]
    return descending, None
  else:
    inverted = invert_rankings(descending, dtype=np.int64)
    descending -= index_offset[None,:]
    inverted -= index_offset[None,:]
    return descending, inverted

def get_query_scores(weights, feature_matrix, qptr, ranking_i):
  return -np.dot(weights.T,feature_matrix[:,qptr[ranking_i]:qptr[ranking_i+1]])

def get_candidate_score_rankings(weights, feature_matrix, qptr, ranking_i, inverted=False):
  scores = -np.dot(weights.T,feature_matrix[:,qptr[ranking_i]:qptr[ranking_i+1]])
  return rank_query(scores,inverted)

def get_candidate_score_ranking(weights,query_feature_matrix,inverted=False):
  scores = -np.dot(weights.T,query_feature_matrix)
  return rank_query(scores,inverted)

def rank_single_query(predictions, inverted=False, n_results=None):
  """
  Given predicted scores of a single query returns rankings.
  """
  ranking = tiebreak_sort(predictions, n_results=n_results)
  if inverted:
    if len(ranking.shape) == 1:
      return invert_ranking(ranking, dtype=np.int32)
    else:
      return invert_rankings(ranking, dtype=np.int32)
  else:
    return ranking

def rank_multiple_queries(predictions, qptr, max_documents=None,
              inverted=False, n_results=None):
  """
  Given predicted scores for queries rankings are generated and returned.
  """

  max_value = np.max(predictions)
  # vector with lenght of each doclist
  n = qptr[1:]-qptr[:-1]
  if not max_documents:
    max_documents = np.max(n)

  # the vector of documents is reshaped into a matrix
  # with a document list on every row
  ind = qptr[:-1,None] + np.arange(0,max_documents)[None,:]
  ind = np.minimum(ind,qptr[1:,None]-1)
  # warped is now a matrix of size n_queries x max_documents
  warped = predictions[ind]
  # every document that appears in a row but not in the query list
  # (due to n_query_list < max_documents) gets the worst score in all documents
  # this makes sure they do not appear in the final ranking
  warped[np.arange(max_documents)[None,:] >= n[:,None]] = max_value + 1

  # tiebreak sort uses numpy to rank every row in the matrix
  # this is faster than ranking them by seperate calls
  rankings = tiebreak_sort(warped, n_results=n_results)
  if inverted:
    inverted = invert_rankings(rankings, dtype=np.int32)
    return inverted[np.arange(max_documents)[None,:] < n[:,None]]
  else:
    return rankings[np.arange(max_documents)[None,:] < n[:,None]]
