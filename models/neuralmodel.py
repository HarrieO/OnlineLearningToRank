import numpy as np

class NeuralModel(object):

  def __init__(self, learning_rate,
               learning_rate_decay,
               hidden_layers, n_features):
    def normal(init, shape):
      return np.random.normal(0., init, shape)

    self.learning_rate = learning_rate
    self.hidden_layer_nodes = hidden_layers
    self.hidden_layers = []
    self.biases = []
    prev_units = n_features
    for n_units in hidden_layers:
      init = 1./prev_units
      self.hidden_layers.append(normal(init, (prev_units, n_units)))
      self.biases.append(normal(init, n_units)[None, :])
      prev_units = n_units
    self.hidden_layers.append(normal(1./prev_units, (prev_units, 1)))
    self.learning_rate_decay = learning_rate_decay

  def score(self, features):
    prev_layer = features
    self.input = features
    self.activations = [prev_layer]
    for hidden_layer, bias in zip(self.hidden_layers[:-1], self.biases):
      prev_layer = np.dot(prev_layer, hidden_layer)
      prev_layer += bias
      prev_layer = 1./(1. + np.exp(-prev_layer))
      self.activations.append(prev_layer)
    result = np.dot(prev_layer, self.hidden_layers[-1])
    self.activations.append(result)
    return result[:, 0]

  def backpropagate(self, doc_ind, doc_weights):
    activations = [a[doc_ind, :] for a in self.activations]
    doc_weights = np.expand_dims(doc_weights, axis=1)
    cur_der = (np.dot(activations[-2].T, doc_weights), None)
    derivatives = [cur_der]
    prev_der = doc_weights
    for i in range(len(self.hidden_layers)-1):
      prev_der = np.dot(prev_der, self.hidden_layers[-i-1].T)
      prev_der *= activations[-i-2]*(1.-activations[-i-2])

      w_der = np.dot(activations[-i-3].T, prev_der)
      b_der = np.sum(prev_der, axis=0, keepdims=True)

      derivatives.append((w_der, b_der))

    return derivatives

  def debugstr(self):
    for i, hd in enumerate(self.hidden_layers[:-1]):
      print 'layer %d:' % i, hd
      print 'bias %d:' % i, self.biases[i]
    print 'final hidden:', self.hidden_layers[-1]


  def update_to_documents(self, doc_ind, doc_weights):
    derivatives = self.backpropagate(doc_ind, doc_weights)

    first_wd = derivatives[0][0]
    self.hidden_layers[-1] += first_wd * self.learning_rate
    for i, (wd, bd) in enumerate(derivatives[1:], 2):
      self.hidden_layers[-i] += wd * self.learning_rate
      self.biases[-i + 1] += bd * self.learning_rate
    self.learning_rate *= self.learning_rate_decay 
    