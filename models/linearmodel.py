import numpy as np

class LinearModel(object):
  def __init__(self, n_features, learning_rate,
               n_candidates=0, learning_rate_decay=1.0):
    self.n_features = n_features
    self.learning_rate = learning_rate
    self.n_models = n_candidates + 1
    self.weights = np.zeros((n_features, self.n_models))
    self.learning_rate_decay = learning_rate_decay

  def copy(self):
    copy = LinearModel(n_features = self.n_features,
                       learning_rate = self.learning_rate,
                       n_candidates = self.n_models-1)
    copy.weights = self.weights.copy()
    return copy

  def candidate_score(self, features):
    self._last_features = features
    return np.dot(features, self.weights).T

  def score(self, features):
    self._last_features = features
    return np.dot(features, self.weights[:,0:1])[:,0]

  def sample_candidates(self):
    assert self.n_models > 1
    vectors = np.random.randn(self.n_features, self.n_models-1)
    vector_norms = np.sum(vectors ** 2, axis=0) ** (1. / 2)
    vectors /= vector_norms[None, :]
    self.weights[:, 1:] = self.weights[:, 0, None] + vectors

  def update_to_mean_winners(self, winners):
    assert self.n_models > 1
    if len(winners) > 0:
      # print 'winners:', winners
      gradient = np.mean(self.weights[:, winners], axis=1) - self.weights[:, 0]
      self.weights[:, 0] += self.learning_rate * gradient
      self.learning_rate *= self.learning_rate_decay

  def update_to_documents(self, doc_ind, doc_weights):
    weighted_docs = self._last_features[doc_ind, :] * doc_weights[:, None]
    gradient = np.sum(weighted_docs, axis=0)
    self.weights[:, 0] += self.learning_rate * gradient
    self.learning_rate *= self.learning_rate_decay

