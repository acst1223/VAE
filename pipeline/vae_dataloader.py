import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.utils import shuffle


def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform', test_ratio=None, CNN_option=False):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        if CNN_option:
            train_neg = train_pos # let size of training negatives equal to that of training positives
        else:
            train_neg = int(train_ratio * x_neg.shape[0])
        if test_ratio:
            test_pos = x_pos.shape[0] - int(test_ratio * x_pos.shape[0])
            if CNN_option:
                test_neg = int(train_neg * 1.33)
            else:
                test_neg = x_neg.shape[0] - int(test_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        if test_ratio:
            x_test = np.hstack([x_pos[test_pos:], x_neg[test_neg:]])
            y_test = np.hstack([y_pos[test_pos:], y_neg[test_neg:]])
            x_validate = np.hstack([x_pos[train_pos:test_pos], x_neg[train_neg:test_neg]])
            y_validate = np.hstack([y_pos[train_pos:test_pos], y_neg[train_neg:test_neg]])
        else:
            x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
            y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        if test_ratio:
            num_test = int(test_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        if test_ratio:
            x_test = x_data[-num_test:]
            x_validate = x_data[num_train: -num_test]
        else:
            x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            if test_ratio:
                y_test = y_data[-num_test:]
                y_validate = y_data[num_train: -num_test]
            else:
                y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    if test_ratio:
        return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)
    else:
        return (x_train, y_train), (x_test, y_test), (None, None)


class DataLoader(object):
    def __init__(self, data_instance_file, train_ratio, test_ratio, h, wipe_train_anomaly=True):
        self.h = h
        pad_to = self.h     # if length of event sequence is less than this value, pad to this value

        # parse data instance file
        data_df = pd.read_csv(data_instance_file)
        data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
        # Split train and test data
        (x_train_raw, self.y_train), \
        (x_test_raw, self.y_test),\
        (x_validate_raw, self.y_validate) = _split_data(data_df['EventSequence'].values,
                                                       data_df['Label'].values,
                                                       train_ratio,
                                                       test_ratio=test_ratio)

        x_temp = x_train_raw
        x_train_raw = []
        for i, y in enumerate(self.y_train):
            if (y == 0 or not wipe_train_anomaly) and len(x_temp[i]) >= h - 2:
                x_train_raw.append(x_temp[i])

        # pad
        x_train_raw = [self._pad(t, pad_to) for t in x_train_raw]
        x_test_raw = [self._pad(t, pad_to) for t in x_test_raw]
        x_validate_raw = [self._pad(t, pad_to) for t in x_validate_raw]

        # process templates
        tot_templates = []
        for item in x_train_raw + x_test_raw + x_validate_raw:
            tot_templates += item
        tot_templates = set(tot_templates)
        if '_PAD' in tot_templates:
            tot_templates.remove('_PAD')
        self.templates = sorted(list(tot_templates))   # Sorting is important!!
        self.template_cnt = len(self.templates)

        self.template_idxs = {}
        for k, template in enumerate(self.templates):
            self.template_idxs[template] = k
        self.template_idxs['_PAD'] = self.template_cnt
        self.vectors = np.zeros((self.template_cnt + 1, self.template_cnt), dtype=np.float64)
        self.vectors[0: self.template_cnt, 0: self.template_cnt] = np.eye(self.template_cnt, dtype=np.float64)

        # print stats
        print('DataLoader: Total templates: ', self.template_cnt)
        print(self.templates)

        # gen data
        self.x_train = self._gen_inputs(x_train_raw)
        self.x_test = self._gen_inputs(x_test_raw)
        self.x_validate = self._gen_inputs(x_validate_raw)

        # gen count
        self.x_train_count = self._get_count_of_sequence(x_train_raw)
        self.x_test_count = self._get_count_of_sequence(x_test_raw)
        self.x_validate_count = self._get_count_of_sequence(x_validate_raw)

    def one_hot_map_flow(self, templates):
        t = self.vectors[templates]
        return [t.reshape(t.shape[0], -1)]

    @staticmethod
    def _pad(target, l):
        return ['_PAD'] * (l - len(target)) + target

    def _get_idx(self, template):
        return self.template_idxs[template]

    def _gen_inputs(self, x):
        inputs = []
        for i in range(len(x)):
            temp_input = []
            for j in range(len(x[i]) - self.h + 1):
                temp_input.append(np.array(list(map(self._get_idx, x[i][j: j + self.h])), dtype=np.uint32))
            temp_input = np.vstack(temp_input)
            inputs.append(temp_input)
        inputs = np.concatenate(inputs)
        return inputs

    def _get_count_of_sequence(self, x):
        '''
        Note that each event sequence is not of the same length. When raw data become arrays
        or tensors, we cannot tell how to split these arrays/tensors back into event sequences
        without information about length of sequence.
        However, note that when window size(h) is not 1 (under most situations it is much larger
        than 1), the length of event sequence is not equal to length of the array generated from
        that sequence. Therefore get count of sequence returns lengths of the array generated
        from each event sequence (which == length of raw event sequence after padding - h + 1).

        Args:
            x: A list of event sequences after padding.

        Returns:
            A list, each element of the list is length of the array generated from
                each event sequence.
        '''
        return [len(t) - self.h + 1 for t in x]
