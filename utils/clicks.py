# -*- coding: utf-8 -*-

import numpy as np


class ClickModel(object):

  '''
  Class for cascading click-models used to simulate clicks.
  '''

  def __init__(self, name, data_type, PCLICK, PSTOP):
    '''
    Name is used for logging, data_type denotes the degrees of relevance the data uses.
    PCLICK and PSTOP the probabilities used by the model.
    '''
    self.name = name
    self.type = data_type
    self.PCLICK = PCLICK
    self.PSTOP = PSTOP

  def get_name(self):
    '''
    Name that can be used for logging.
    '''
    return self.name + '_' + self.type

  def generate_clicks(self, ranking, all_labels):
    '''
    Generates clicks for a given ranking and relevance labels.
    ranking: np array of indices which correspond with all_labels
    all_labels: np array of integers
    '''
    labels = all_labels[ranking]
    coinflips = np.random.rand(*ranking.shape)
    clicks = coinflips < self.PCLICK[labels]
    coinflips = np.random.rand(*ranking.shape)
    stops = coinflips < self.PSTOP[labels]
    stopped_clicks = np.zeros(ranking.shape, dtype=bool)
    if np.any(stops):
        clicks_before_stop = np.logical_and(clicks, np.arange(ranking.shape[0])
                                            <= np.where(stops)[0][0])
        stopped_clicks[clicks_before_stop] = True
        return stopped_clicks
    else:
        return np.zeros(ranking.shape, dtype=bool) + clicks

class ExamineClickModel(object):

  '''
  Class for cascading click-models used to simulate clicks.
  '''

  def __init__(self, name, data_type, PCLICK, eta):
    '''
    Name is used for logging, data_type denotes the degrees of relevance the data uses.
    PCLICK and PSTOP the probabilities used by the model.
    '''
    self.name = name
    self.type = data_type
    self.PCLICK = PCLICK
    self.eta = eta

  def get_name(self):
    '''
    Name that can be used for logging.
    '''
    return self.name + '_' + self.type

  def generate_clicks(self, ranking, all_labels):
    '''
    Generates clicks for a given ranking and relevance labels.
    ranking: np array of indices which correspond with all_labels
    all_labels: np array of integers
    '''
    n_results = ranking.shape[0]
    examine_prob = (1./(np.arange(n_results)+1))**self.eta
    stop_prob = np.ones(n_results)
    stop_prob[1:] -= examine_prob[1:]/examine_prob[:-1]
    stop_prob[0] = 0.

    labels = all_labels[ranking]
    coinflips = np.random.rand(*ranking.shape)
    clicks = coinflips < self.PCLICK[labels]
    coinflips = np.random.rand(n_results)
    stops = coinflips < stop_prob
    stops = np.logical_and(stops, clicks)
    stopped_clicks = np.zeros(ranking.shape, dtype=bool)
    if np.any(stops):
        clicks_before_stop = np.logical_and(clicks, np.arange(ranking.shape[0])
                                            <= np.where(stops)[0][0])
        stopped_clicks[clicks_before_stop] = True
        return stopped_clicks
    else:
        return np.zeros(ranking.shape, dtype=bool) + clicks


# create synonyms for keywords to ease command line use
syn_tuples = [
    ('ex_per_1', ['exper1']),
    ('navigational', ['nav', 'navi', 'navig', 'navigat']),
    ('informational', ['inf', 'info', 'infor', 'informat']),
    ('perfect', ['per', 'perf']),
    ('almost_random', [
        'alm',
        'almost',
        'alra',
        'arand',
        'almostrandom',
        'almrand',
        ]),
    ('random', ['ran', 'rand']),
    ('binary', ['bin']),
    ('short', []),
    ('long', []),
    ]
synonyms = {}
for full, abrv_list in syn_tuples:
    assert full not in synonyms or synonyms[full] == full
    synonyms[full] = full
    for abrv in abrv_list:
        assert abrv not in synonyms or synonyms[abrv] == full
        synonyms[abrv] = full

bin_models = {}
bin_models['navigational'] = np.array([.05, .95]), np.array([.2, .9])
bin_models['informational'] = np.array([.4, .9]), np.array([.1, .5])
bin_models['perfect'] = np.array([.0, 1.]), np.array([.0, .0])
bin_models['almost_random'] = np.array([.4, .6]), np.array([.5, .5])
bin_models['random'] = np.array([.5, .5]), np.array([.0, .0])
bin_models['ex_per_1'] = np.array([.0, 1.]), 1.0

short_models = {}
short_models['navigational'] = np.array([.05, .5, .95]), np.array([.2, .5, .9])
short_models['informational'] = np.array([.4, .7, .9]), np.array([.1, .3, .5])
short_models['perfect'] = np.array([.0, .5, 1.]), np.array([.0, .0, .0])
short_models['almost_random'] = np.array([.4, .5, .6]), np.array([.5, .5, .5])
short_models['random'] = np.array([.5, .5, .5]), np.array([.0, .0, .0])
short_models['ex_per_1'] = np.array([.0, .5, 1.]), 1.0

long_models = {}
long_models['navigational'] = np.array([.05, .3, .5, .7, .95]), np.array([.2, .3, .5, .7, .9])
long_models['informational'] = np.array([.4, .6, .7, .8, .9]), np.array([.1, .2, .3, .4, .5])
long_models['perfect'] = np.array([.0, .2, .4, .8, 1.]), np.array([.0, .0, .0, .0, .0])
long_models['almost_random'] = np.array([.4, .45, .5, .55, .6]), np.array([.5, .5, .5, .5, .5])
long_models['random'] = np.array([.5, .5, .5, .5, .5]), np.array([.0, .0, .0, .0, .0])
long_models['ex_per_1'] = np.array([.0, .2, .4, .8, 1.]), 1.0

all_models = {'short': short_models, 'binary': bin_models, 'long': long_models}

def get_click_models(keywords):
    '''
  Convenience function which returns click models corresponding with keywords.
  only returns click functions for one data type: (bin,short,long)
  '''
    type_name = None
    type_keyword = None
    for keyword in keywords:
        assert keyword in synonyms
        if synonyms[keyword] in all_models:
            type_name = synonyms[keyword]
            type_keyword = keyword
            break
    assert type_name is not None and type_keyword is not None

    models_type = all_models[type_name]
    full_names = [synonyms[key] for key in keywords if key != type_keyword]

    click_models = []
    for full in full_names:
        if full == 'ex_per_1':
            c_m = ExamineClickModel(full, type_name, *models_type[full])
        else:
            c_m = ClickModel(full, type_name, *models_type[full])
        click_models.append(c_m)

    return click_models
