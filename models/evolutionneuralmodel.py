import numpy as np

class EvolutionNeuralModel(object):

  def __init__(self, learning_rate,
               learning_rate_decay,
               hidden_layers, n_features,
               n_candidates):
    def normal(init, shape):
      safe_shape = (self.n_models,) + shape
      return np.random.normal(0., init, safe_shape)

    self.n_models = n_candidates + 1
    self.learning_rate = learning_rate
    self.hidden_layer_nodes = hidden_layers
    self.hidden_layers = []
    self.biases = []
    self.n_nodes = 0
    prev_units = n_features
    for n_units in hidden_layers:
      init = 1./prev_units
      self.hidden_layers.append(normal(init, (prev_units, n_units)))
      self.biases.append(normal(init, (1, n_units,)))
      self.n_nodes += (prev_units+1)*n_units
      prev_units = n_units
    self.hidden_layers.append(normal(1./prev_units, (prev_units, 1)))
    self.n_nodes += prev_units
    self.learning_rate_decay = learning_rate_decay

  def sample_candidates(self):
    assert self.n_models > 1
    n_cand = self.n_models-1
    vectors = np.random.randn(self.n_models-1, self.n_nodes)
    vector_norms = np.sum(vectors ** 2, axis=1) ** (1. / 2)
    vectors /= vector_norms[:, None]
    vec_i = 0
    for hidden_layer, bias in zip(self.hidden_layers[:-1], self.biases):
      h_shape = hidden_layer.shape[1:3]
      n_matrix = np.prod(h_shape)
      n_bias = h_shape[1]
      matrix_noise = np.reshape(vectors[:, vec_i:vec_i+n_matrix],
                                (n_cand, h_shape[0], h_shape[1]))
      vec_i += n_matrix
      bias_noise = np.reshape(vectors[:, vec_i:vec_i+n_bias],
                              (n_cand, n_bias))
      vec_i += n_bias

      hidden_layer[1:,:,:] = hidden_layer[0, None,:,:] + matrix_noise
      bias[1:, :] = bias[0, None, :] + bias_noise

    matrix_noise = vectors[:,vec_i:,None]
    self.hidden_layers[-1][1:,:,:] = self.hidden_layers[-1][0,None,:,:] + matrix_noise

  def score(self, features):
    return self._score(features, 0)

  def _score(self, features, model_i):
    prev_layer = features
    self.input = features
    self.activations = [prev_layer]
    for hidden_layer, bias in zip(self.hidden_layers[:-1], self.biases):
      prev_layer = np.dot(prev_layer, hidden_layer[model_i, :])
      prev_layer += bias[model_i,:]
      prev_layer = 1./(1. + np.exp(-prev_layer))
      self.activations.append(prev_layer)
    result = np.dot(prev_layer, self.hidden_layers[-1][model_i,: ])
    self.activations.append(result)
    return result[:, 0] 

  def candidate_score(self, features):
    scores = []
    for i in range(self.n_models):
      scores.append(self._score(features, i))
    return np.stack(scores, axis=0)

  def update_to_mean_winners(self, winners):
    assert self.n_models > 1
    if len(winners) > 0:
      for hidden_layer, bias in zip(self.hidden_layers[:-1], self.biases):
        average_layer = np.mean(hidden_layer[winners,:,:], axis=0)
        average_bias = np.mean(bias[winners,:], axis=0)

        layer_gradient = (average_layer - hidden_layer[0,:,:])
        bias_gradient = (average_bias - bias[0,:])

        hidden_layer[0,:,:] += self.learning_rate*layer_gradient
        bias[0,:] += self.learning_rate*bias_gradient
      
      average_layer = np.mean(self.hidden_layers[-1][winners,:,:], axis=0)
      layer_gradient = (average_layer - self.hidden_layers[-1][0,:,:])
      self.hidden_layers[-1][0,:,:] += self.learning_rate*layer_gradient

      self.learning_rate *= self.learning_rate_decay
    