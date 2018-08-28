import numpy as np

class NeuralModel(object):

  def __init__(self, learning_rate, hidden_layers, n_features,
               regularize_rate=0., n_output=1):
    self.learning_rate = learning_rate
    self.regularize_rate = regularize_rate
    self.hidden_layer_nodes = hidden_layers
    self.hidden_layers = []
    self.biases = []
    prev_units = n_features
    for n_units in hidden_layers:
      self.hidden_layers.append(np.random.normal(0., 1./prev_units, (prev_units, n_units)))
      self.biases.append(np.random.normal(0., 1./prev_units, n_units)[None, :])
      prev_units = n_units
    self.hidden_layers.append(np.random.normal(0., 1./prev_units, (prev_units, n_output)))

  def score(self, input):
    prev_layer = input.T
    self.input = input
    self.activations = [prev_layer]
    for hidden_layer, bias in zip(self.hidden_layers[:-1], self.biases):
      prev_layer = np.dot(prev_layer, hidden_layer)
      prev_layer += bias
      # prev_layer = np.maximum(0., prev_layer)
      prev_layer = 1./(1. + np.exp(-prev_layer))
      self.activations.append(prev_layer)
    result = np.dot(prev_layer, self.hidden_layers[-1])
    self.activations.append(result)
    return result

  # def predict(self, input):
  #   prev_layer = input.T
  #   for hidden_layer, bias in zip(self.hidden_layers[:-1], self.biases):
  #     prev_layer = np.dot(prev_layer, hidden_layer)
  #     prev_layer += bias
  #     # prev_layer = np.maximum(0, prev_layer)
  #     prev_layer = 1./(1. + np.exp(-prev_layer))
  #   return np.dot(prev_layer, self.hidden_layers[-1])

  def backpropagate(self, doc_weights):
    activations = self.activations
    doc_weights = np.expand_dims(doc_weights, axis=1)
    cur_der = (np.dot(activations[-2].T, doc_weights), None)
    derivatives = [cur_der]
    prev_der = doc_weights
    for i in range(len(self.hidden_layers)-1):
      prev_der = np.dot(prev_der, self.hidden_layers[-i-1].T)
      # prev_der[activations[-i-2] <= 0] = 0
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


  def update_to_documents(self, doc_weights):
    derivatives = self.backpropagate(doc_weights)

    first_wd = derivatives[0][0]
    self.hidden_layers[-1] += first_wd * self.learning_rate
    for i, (wd, bd) in enumerate(derivatives[1:], 2):
      self.hidden_layers[-i] += wd * self.learning_rate
      self.biases[-i + 1] += bd * self.learning_rate


  # def regularize_update(self):
  #   rate = self.regularize_rate
  #   if rate != 0:
  #     self.hidden_layers[-1] -= rate * self.hidden_layers[-1]
  #     for i in range(len(self.hidden_layers) - 1):
  #       self.hidden_layers[i] -= rate * self.hidden_layers[i]
  #       # self.biases[i] -= rate * self.biases[i] * 0.1

    