# -*- coding: utf-8 -*-

import sharedmem
import glob
import numpy as np
import os.path
import gc
import math
import cPickle

FOLDDATA_WRITE_VERSION = 3

def _add_zero_to_vector(vector, dtype=np.int32):
    return np.concatenate([np.zeros(1, dtype=dtype), vector])

class DataSet(object):

    """
    Class designed to manage meta-data for datasets.
    """

    def __init__(self, name, data_paths, click_model_type, small=False, num_features=None,
                 output_dir=None, max_folds=-1, multileave_feat=None):
        self.name = name
        self.small = small
        self.click_model_type = click_model_type
        self.num_features = num_features
        if output_dir is None:
            self.output_dir = name
        else:
            self.output_dir = output_dir
        if type(data_paths) == str:
            self.data_paths = glob.glob(data_paths)
        elif type(data_paths) == list:
            self.data_paths = []
            for path in data_paths:
                self.data_paths.extend(glob.glob(path))
        else:
            assert False, 'Unknown type for data_paths: %s' % str(type(data_paths))
        if max_folds <= 0:
            self.max_folds = 999
        else:
            self.max_folds = max_folds

        if multileave_feat:
            self._multileave_feat = []
            for feat in multileave_feat:
                if type(feat) == str:
                    self._multileave_feat.append(feat)
                elif type(feat) == int:
                    self._multileave_feat.append(str(feat))
                elif type(feat) == list:
                    assert all(type(x) in [int,str] for x in feat), 'Tried to add non int feature in list for  %s.' % self.name
                    self._multileave_feat.extend([str(x) for x in feat])
                else:
                    assert False, "Invalid feature type %s given for dataset %s." % (type(feat), self.name)
        else:
            self._multileave_feat = [str(x) for x in range(1,self.num_features+1)]

    def num_data_folds(self):
        return len(self.data_paths)

    def get_data_folds(self, sim_args):
        for i, path in enumerate(self.data_paths):
            yield DataFold(sim_args, self, i, path)

    def num_runs_per_fold(self, sim_args):
        return int(math.ceil(sim_args.n_runs / float(self.num_data_folds())))

    def multileave_feat(self):
        return self._multileave_feat

class DataFold(object):

    def __init__(self, sim_args, dataset, fold_num, data_path):
        self.name = dataset.name
        self.max_folds = dataset.max_folds
        self.click_model_type = dataset.click_model_type
        self.small = dataset.small
        self.num_features = dataset.num_features
        self.fold_num = fold_num
        self.data_path = data_path
        self._sim_args = sim_args
        self._data_ready = False
        self.validation_data = sim_args.validation
        self.num_folds = dataset.num_data_folds()
        self.num_runs_per_fold = dataset.num_runs_per_fold(sim_args)
        self._raw_multileave_feat = dataset.multileave_feat()
        if not self.validation_data:
            self.heldout_tag = 'TEST'
        else:
            self.heldout_tag = 'VALI'

    def train_query_labels(self, ranking_index):
      s_i = self.train_doclist_ranges[ranking_index]
      e_i = self.train_doclist_ranges[ranking_index+1]
      return self.train_label_vector[s_i:e_i]

    def test_query_labels(self, ranking_index):
      s_i = self.test_doclist_ranges[ranking_index]
      e_i = self.test_doclist_ranges[ranking_index+1]
      return self.test_label_vector[s_i:e_i]

    def n_train_queries(self):
      return self.train_doclist_ranges.shape[0] - 1

    def n_train_docs(self):
      return self.train_feature_matrix.shape[0]

    def n_test_queries(self):
      return self.test_doclist_ranges.shape[0] - 1

    def n_test_docs(self):
      return self.test_feature_matrix.shape[0]

    def num_features_known(self):
        return not self.num_features is None

    def data_ready(self):
        return self._data_ready

    def get_multileave_feat(self):
        assert self.data_ready, 'Unable to get multileave_feat for %s before data is read.' % self.name
        return np.array([self.feature_map[x] for x in self._raw_multileave_feat if x in self.feature_map])

    def clean_data(self):
        del self.train_feature_matrix
        del self.train_doclist_ranges
        del self.train_label_vector
        del self.test_feature_matrix
        del self.test_doclist_ranges
        del self.test_label_vector
        self._data_ready = False
        gc.collect()

    def _make_shared(self, numpy_matrix):
        """
        Avoids the copying of Read-Only shared memory.
        """
        if self._sim_args.n_processing == 1:
            return numpy_matrix
        if numpy_matrix is None:
            return None
        shared = sharedmem.empty(numpy_matrix.shape, dtype=numpy_matrix.dtype)
        shared[:] = numpy_matrix[:]
        return shared

    def _read_file(self, path, all_features=None, filter_non_uniq=False):
        '''
        Read letor file and returns dict for qid to indices, labels for queries
        and list of doclists of features per doc per query.
        '''
        current_qid = None
        queries = {}
        queryIndex = 0
        doclists = []
        labels = []
        if all_features is None:
            all_features = {}
            features_to_keep = {}
        else:
            features_to_keep = all_features.copy()

        featureMax = {}
        featureMin = {}
        for line in open(path, 'r'):
            info = line[:line.find('#')].split()

            qid = info[1].split(':')[1]
            label = int(info[0])
            if qid not in queries:
                queryIndex = len(queries)
                queries[qid] = queryIndex
                doclists.append([])
                labels.append([])
                current_qid = qid
            elif qid != current_qid:
                queryIndex = queries[qid]
                current_qid = qid

            featureDict = {}
            for pair in info[2:]:
                featid, feature = pair.split(':')
                all_features[featid] = True
                feat_value = float(feature)
                featureDict[featid] = feat_value
                if featid in featureMax:
                    featureMax[featid] = max(featureMax[featid], feat_value)
                    featureMin[featid] = min(featureMin[featid], feat_value)
                else:
                    featureMax[featid] = feat_value
                    featureMin[featid] = feat_value
            doclists[queryIndex].append(featureDict)
            labels[queryIndex].append(label)

        if filter_non_uniq:
            unique_features = {}
            for featid in all_features:
                if featid in features_to_keep:
                    unique_features[featid] = True
                elif featureMax[featid] > featureMin[featid]:
                    unique_features[featid] = True
            return queries, doclists, labels, unique_features
        else:
            return queries, doclists, labels, all_features

    def _create_feature_mapping(self, feature_dict):
        total_features = 0
        feature_map = {}
        for fid in feature_dict:
            if fid not in feature_map:
                feature_map[fid] = total_features
                total_features += 1
        return feature_map

    def _convert_featureDicts(self, doclists, label_lists, feature_mapping, query_level_norm=True):
        """
        represents doclists/features as matrix and list of ranges
        """
        if self.num_features_known:
            total_features = self.num_features
        else:
            total_features = len(feature_mapping)
        total_docs = 0
        ranges = []
        for doclist in doclists:
            start_range = total_docs
            total_docs += len(doclist)
            ranges.append((start_range, total_docs))

        feature_matrix = np.zeros((total_features, total_docs))
        label_vector = np.zeros(total_docs, dtype=np.int32)

        new_doclists = None
        index = 0
        unique_values = np.zeros(total_features, dtype=bool)
        for doclist, labels in zip(doclists, label_lists):
            start = index
            for featureDict, label in zip(doclist, labels):
                for fid, value in featureDict.items():
                    if fid in feature_mapping:
                        feature_matrix[feature_mapping[fid], index] = value
                label_vector[index] = label
                index += 1
            end = index
            if query_level_norm:
                feature_matrix[:, start:end] -= np.amin(feature_matrix[:, start:end], axis=1)[:,
                        None]
                safe_max = np.amax(feature_matrix[:, start:end], axis=1)
                safe_ind = safe_max != 0
                feature_matrix[safe_ind, start:end] /= safe_max[safe_ind][:, None]
                unique_values = np.logical_or(unique_values, safe_ind)

        qptr = np.zeros(len(ranges) + 1, dtype=np.int32)
        for i, ra in enumerate(ranges):
            qptr[i + 1] = ra[1]

        return self._make_shared(feature_matrix), self._make_shared(qptr), \
            self._make_shared(label_vector)

    def read_data(self):
        """
        Reads data from a fold folder (letor format).
        """
        # clear any previous datasets
        gc.collect()

        validation_in_train = self._sim_args.validation_in_train
        validation_as_test = self._sim_args.validation
        train_only = self._sim_args.train_only
        store_pickle_after_read = self._sim_args.store_binarized_data_after_read
        read_from_pickle = self._sim_args.read_binarized_data

        train_read = False
        test_read = False
        fmap_read = False
        if validation_as_test:
            train_pickle_name = 'binarized_train_val.npz'
            test_pickle_name = 'binarized_val.npz'
        elif not validation_in_train:
            train_pickle_name = 'binarized_train_no_val.npz'
            test_pickle_name = 'binarized_test.npz'
        else:
            train_pickle_name = 'binarized_train.npz'
            test_pickle_name = 'binarized_test.npz'

        train_pickle_path = self.data_path + train_pickle_name
        test_pickle_path = self.data_path + test_pickle_name
        fmap_pickle_path = self.data_path + 'binarized_fmap.pickle'
        if read_from_pickle:
            if os.path.isfile(fmap_pickle_path):
                with open(fmap_pickle_path, 'rb') as f:
                    loaded = cPickle.load(f)
                    if loaded[0] == FOLDDATA_WRITE_VERSION:
                        self.feature_map = loaded[1]
                        fmap_read = True
            if os.path.isfile(train_pickle_path):
                loaded_data = np.load(train_pickle_path)
                del loaded_data.f
                if 'train_version' in loaded_data and loaded_data['train_version'] \
                    == FOLDDATA_WRITE_VERSION:
                    self.train_feature_matrix = self._make_shared(loaded_data['feature_matrix'])
                    self.train_doclist_ranges = self._make_shared(loaded_data['doclist_ranges'])
                    self.train_label_vector = self._make_shared(loaded_data['label_vector'])
                    train_read = True
                del loaded_data
                gc.collect()
            if os.path.isfile(test_pickle_path):
                loaded_data = np.load(test_pickle_path)
                del loaded_data.f
                if 'test_version' in loaded_data and loaded_data['test_version'] \
                    == FOLDDATA_WRITE_VERSION:
                    self.test_feature_matrix = self._make_shared(loaded_data['test_feature_matrix'])
                    self.test_doclist_ranges = self._make_shared(loaded_data['test_doclist_ranges'])
                    self.test_label_vector = self._make_shared(loaded_data['test_label_vector'])
                    test_read = True
                # remove potentially memory intensive variables
                del loaded_data
                gc.collect()
        # without a feature map we can't guarantee read train or test data will be compatible
        # so everything must be reread from binary
        if not fmap_read and not (train_read and test_read):
            train_read = False
            test_read = False

        if not train_read:
            doclists = []
            labels = []
            _, n_doclists, n_labels, training_features = self._read_file(self.data_path
                    + 'train.txt', filter_non_uniq=True)
            doclists.extend(n_doclists)
            labels.extend(n_labels)

            if not validation_as_test and validation_in_train:
                _, n_doclists, n_labels, training_features = self._read_file(self.data_path
                        + 'vali.txt', training_features, filter_non_uniq=True)
                doclists.extend(n_doclists)
                labels.extend(n_labels)

            if not fmap_read:
                self.feature_map = self._create_feature_mapping(training_features)
                with open(fmap_pickle_path, 'wb') as f:
                    cPickle.dump((FOLDDATA_WRITE_VERSION, self.feature_map), f)

            self.train_feature_matrix, self.train_doclist_ranges, self.train_label_vector = \
                self._convert_featureDicts(doclists, labels, self.feature_map)
            del doclists
            del labels
            # invoking garbage collection, to avoid memory clogging
            gc.collect()
            if store_pickle_after_read:
                np.savez(train_pickle_path, train_version=FOLDDATA_WRITE_VERSION,
                         feature_map=self.feature_map, feature_matrix=self.train_feature_matrix,
                         doclist_ranges=self.train_doclist_ranges,
                         label_vector=self.train_label_vector)

        if not train_only and not test_read:
            if not validation_as_test:
                _, test_doclists, test_labels, _ = self._read_file(self.data_path + 'test.txt')
            else:
                _, test_doclists, test_labels, _ = self._read_file(self.data_path + 'vali.txt')

            self.test_feature_matrix, self.test_doclist_ranges, self.test_label_vector = \
                self._convert_featureDicts(test_doclists, test_labels, self.feature_map)
            del test_doclists
            del test_labels
            # invoking garbage collection, to avoid memory clogging
            gc.collect()
            if store_pickle_after_read:
                np.savez(test_pickle_path, test_version=FOLDDATA_WRITE_VERSION,
                         test_feature_matrix=self.test_feature_matrix,
                         test_doclist_ranges=self.test_doclist_ranges,
                         test_label_vector=self.test_label_vector)

        elif train_only:
            self.test_feature_matrix = None
            self.test_doclist_ranges = None
            self.test_label_vector = None

        if not train_only and self._sim_args.purge_test_set:
            n_queries = self.test_label_vector.shape[0]
            cum_label = np.cumsum(self.test_label_vector)[self.test_doclist_ranges[1:]-1]
            cum_label = _add_zero_to_vector(cum_label)
            n_labels = cum_label[1:]-cum_label[:-1]
            if np.any(n_labels == 0):
                cum_q = (n_labels > 0).astype(np.int32)
                cum_q = np.cumsum(cum_q)
                diff_q = _add_zero_to_vector(cum_q[n_labels == 0][:-1])
                fix_q = diff_q - cum_q[n_labels == 0]
               
                pre_mask = np.ones(n_labels.shape)
                pre_mask[n_labels == 0] = fix_q
                
                mask = np.zeros(n_queries)
                mask[self.test_doclist_ranges[:-1]] = pre_mask
                mask = np.cumsum(mask).astype(bool)
                
                n_docs = self.test_doclist_ranges[1:] - self.test_doclist_ranges[:-1]
                masked_n_docs = n_docs[n_labels > 0]
                self.test_doclist_ranges = _add_zero_to_vector(np.cumsum(masked_n_docs))

                self.test_label_vector = self.test_label_vector[mask]
                self.test_feature_matrix = self.test_feature_matrix[:, mask]

                assert self.test_doclist_ranges[-1] == self.test_label_vector.shape[0]
                assert self.test_feature_matrix.shape[1] == self.test_label_vector.shape[0]

        if not self.num_features_known():
            self.num_features = self.train_feature_matrix.shape[0]
        assert self.num_features == self.train_feature_matrix.shape[0], \
            'Expected %d features but found %d in training matrix' % (self.num_features,
                self.train_feature_matrix.shape[0])
        if not train_only:
            assert self.num_features == self.test_feature_matrix.shape[0], \
                'Expected %d features but found %d in test matrix' % (self.num_features,
                    self.test_feature_matrix.shape[0])

        self.train_feature_matrix = self.train_feature_matrix.T
        self.test_feature_matrix = self.test_feature_matrix.T
        self._data_ready = True
