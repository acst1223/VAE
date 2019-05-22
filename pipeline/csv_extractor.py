# STEP 1

import pandas as pd
import os
import numpy as np
import re
import sys
from sklearn.utils import shuffle
from collections import OrderedDict
from ast import literal_eval
import datetime


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


def _split_idxs(y_data, train_ratio, test_ratio=None, CNN_option=False):
    if not test_ratio:
        test_ratio = 1 - train_ratio
    x_data = np.array(list(range(y_data.shape[0])))
    pos_idx = y_data > 0
    x_pos = x_data[pos_idx]
    x_neg = x_data[~pos_idx]
    train_pos = int(train_ratio * x_pos.shape[0])
    if CNN_option:
        train_neg = train_pos  # let size of training negatives equal to that of training positives
    else:
        train_neg = int(train_ratio * x_neg.shape[0])
    test_pos = x_pos.shape[0] - int(test_ratio * x_pos.shape[0])
    if CNN_option:
        test_neg = int(train_neg * 1.33)
    else:
        test_neg = x_neg.shape[0] - int(test_ratio * x_neg.shape[0])
    x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
    x_test = np.hstack([x_pos[test_pos:], x_neg[test_neg:]])
    x_validate = np.hstack([x_pos[train_pos:test_pos], x_neg[train_neg:test_neg]])
    return x_train, x_test, x_validate # remember here x: idxs


def _split_col_header(st):
    cnt = 0 # col header count
    headers = []
    p = re.compile(r'^\d+\.')
    while True:
        pos = st.find('%d.' % (cnt + 2))
        if pos == -1:
            if p.match(st):
                cnt += 1
                p.sub('', st)
                headers.append(st)
            break
        cnt += 1
        headers.append(st[: pos].strip())
        st = st[pos:]

    # wipe numberings
    for i in range(len(headers)):
        while headers[i][0] != '.':
            headers[i] = headers[i][1:]
        headers[i] = headers[i][1:]

    return (headers)


def _load_HDFS(log_file, col_header_file=None, label_file=None, window='session', train_ratio=0.5, split_type='sequential',
    save_csv=False, is_data_instance=False, test_ratio=None, CNN_option=False):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        is_data_instance: whether the file is data instance, if true, data in data instance will be returned
        test_ratio: ratio of test set(if it is none then there is no validation set)
        CNN_option: if it is True, then split process will be different

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
        (x_validate, y_validate): the validation data
    """

    if CNN_option:
        split_type = 'uniform'

    if is_data_instance:
        assert log_file.endswith('csv')  # data instance file must end with csv
        data_df = pd.read_csv(log_file)
        data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
        # Split train and test data
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                                                           data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(x_data, y_data, train_ratio, split_type,
                                                                                     test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."  # window=session: grouped by EventId
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            if idx % 100000 == 0:
                print("%d rows processed" % idx)
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

        if save_csv:
            data_df.sample(frac=1).to_csv('HDFS/data_instances.csv', index=False)

        if not label_file:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _), (x_validate, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type,
                                                                     test_ratio=test_ratio, CNN_option=CNN_option)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_data, None), (x_test, None), (x_validate, None)

    # Warning: Under such circumstances only data instance files will be generated,
    # then the program will quit
    elif log_file.endswith('.log') or log_file.endswith('.txt'):
        if not col_header_file:
            raise FileNotFoundError("Col header file not found!")

        idx = 0
        output_file = log_file[: -3] + 'csv'

        fw = open(output_file, 'w')
        fw.write('LineId,Date,Time,Pid,Level,Component,Content,EventId,EventTemplate\n')
        fw.close()

        f = open(col_header_file, 'r')
        text = f.readline()[:-1]
        headers = _split_col_header(text)
        patterns = [re.compile(t) for t in headers]  # patterns according to headers
        f.close()

        f = open(log_file, 'r')
        st = '' # buffer

        while True:
            l = f.readline()[:-1].split(' ')
            if not l or len(l) < 6:
                break
            idx += 1
            st += '%d,' % idx  # LineId
            st += '%s,' % l[0]  # Date
            st += '%s,' % l[1]  # Time
            st += '%s,' % l[2]  # Pid
            st += '%s,' % l[3]  # Level
            st += '%s,' % l[4][: -1]  # Component
            content = ' '.join(l[5:])
            posp = content.find(',')
            if posp != -1:
                content = content[: posp]
            for i in range(len(headers)):
                m = patterns[i].match(content)
                if m:
                    st += '%s,' % content  # Content
                    st += 'E%d,' % i  # EventId
                    st += '%s' % headers[i]  # EventTemplate
                    break
            st += '\n'
            if idx % 10000 == 0:
                fw = open(output_file, 'a')
                fw.write(st)
                fw.close()
                st = ''
                print('%d logs converted' % idx)

        f.close()

        if idx % 10000 != 0:
            fw = open(output_file, 'a')
            fw.write(st)
            fw.close()
            st = ''
            print('%d logs converted' % idx)

        return

    else:
        raise NotImplementedError('load_HDFS() only support csv, npz, log and txt files!')

    if test_ratio:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_validate = x_validate.shape[0]
        num_total = num_train + num_test + num_validate
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_validate_pos = sum(y_validate)
        num_pos = num_train_pos + num_test_pos + num_validate_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))
        print('Validate: {} instances, {} anomaly, {} normal\n' \
              .format(num_validate, num_validate_pos, num_validate - num_validate_pos))

    else:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_total = num_train + num_test
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_pos = num_train_pos + num_test_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)


def load_OpenStack(log_file, col_header_file=None, label_file=None, window='session', train_ratio=0.5, split_type='sequential',
    save_csv=False, is_data_instance=False, test_ratio=None, CNN_option=False):
    """ Load OpenStack structured log into train, test and validate data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        is_data_instance: whether the file is data instance, if true, data in data instance will be returned
        test_ratio: ratio of test set(if it is none then there is no validation set)
        CNN_option: if it is True, then split process will be different

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
        (x_validate, y_validate): the validation data
    """

    if CNN_option:
        split_type = 'uniform'

    if is_data_instance:
        assert log_file.endswith('csv')  # data instance file must end with csv
        data_df = pd.read_csv(log_file)
        data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
        # Split train and test data
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                                                           data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(x_data, y_data, train_ratio, split_type,
                                                                                     test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for OpenStack dataset."  # window=session: grouped by EventId
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            if idx % 100000 == 0:
                print("%d rows processed" % idx)
            blkId_list = re.findall(r'.*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}).*', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:
            data_df['Label'] = 0
            fl = open(label_file, 'r')
            while True:
                st = fl.readline()
                if not st:
                    break
                st = st[: -1]
                data_df.ix[data_df['BlockId'] == st, 'Label'] = 1
            fl.close()

            # Split train and test data
            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if not label_file:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _), (x_validate, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type,
                                                                     test_ratio=test_ratio, CNN_option=CNN_option)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_data, None), (x_test, None), (x_validate, None)

    # Warning: Under such circumstances only data instance files will be generated,
    # then the program will quit
    elif log_file.endswith('.log') or log_file.endswith('.txt'):
        if not col_header_file:
            raise FileNotFoundError("Col header file not found!")

        idx = 0
        output_file = log_file[: -3] + 'csv'

        fw = open(output_file, 'w')
        fw.write('LineId,Logrecord,Date,Time,Pid,Level,Component,ADDR,Content,EventId,EventTemplate\n')
        fw.close()

        col_df = pd.read_csv(col_header_file)
        raw_headers = col_df['EventTemplate'].tolist()
        headers = [t.replace(r'[', r'\[') for t in raw_headers]
        headers = [t.replace(r']', r'\]') for t in headers]
        headers = [t.replace(r'.', r'\.') for t in headers]
        headers = [t.replace(r'(', r'\(') for t in headers]
        headers = [t.replace(r')', r'\)') for t in headers]
        headers = [t.replace(r'<*>', r'(.*)') for t in headers]
        patterns = [re.compile(t) for t in headers]  # patterns according to headers

        f = open(log_file, 'r')
        st = '' # buffer

        while True:
            st2 = '' # buffer in loop
            l = f.readline()
            if not l:
                break
            l = l[: -1].split(' ')
            if len(l) < 8:
                print("Warning: Invalid format before index %d" % idx)
                continue
            idx += 1
            st2 += '%d,' % idx  # LineId
            st2 += '%s,' % l[0]  # Logrecord
            st2 += '%s,' % l[1]  # Date
            st2 += '%s,' % l[2]  # Time
            st2 += '%s,' % l[3]  # Pid
            st2 += '%s,' % l[4]  # Level
            st2 += '%s,' % l[5]  # Component
            left = ' '.join(l[6:])
            p = left.find(']')
            if p == -1:
                idx -= 1
                print("Warning: Invalid ADDR before index %d" % idx)
                continue
            st2 += '%s,' % left[1: p] # ADDR
            content = left[p + 2:]
            hit = False
            for i in range(len(headers)):
                m = patterns[i].match(content)
                if m:
                    if content.find(',') == -1:
                        st2 += '%s,' % content  # Content
                    else:
                        st2 += '"%s",' % content  # Content
                    st2 += 'E%d,' % i  # EventId
                    if content.find(',') == -1:
                        st2 += '%s' % raw_headers[i]  # EventTemplate
                    else:
                        st2 += '"%s"' % raw_headers[i]  # EventTemplate
                    hit = True
                    break
            if not hit:
                idx -= 1
                continue
            st2 += '\n'
            st += st2
            if idx % 10000 == 0:
                fw = open(output_file, 'a')
                fw.write(st)
                fw.close()
                st = ''
                print('%d logs converted' % idx)

        f.close()

        if idx % 10000 != 0:
            fw = open(output_file, 'a')
            fw.write(st)
            fw.close()
            st = ''
            print('%d logs converted' % idx)

        exit(0)

    else:
        raise NotImplementedError('load_OpenStack() only support csv, npz, log and txt files!')

    if test_ratio:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_validate = x_validate.shape[0]
        num_total = num_train + num_test + num_validate
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_validate_pos = sum(y_validate)
        num_pos = num_train_pos + num_test_pos + num_validate_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))
        print('Validate: {} instances, {} anomaly, {} normal\n' \
              .format(num_validate, num_validate_pos, num_validate - num_validate_pos))

    else:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_total = num_train + num_test
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_pos = num_train_pos + num_test_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)


def _load_HDFS_instances(log_file, train_ratio, test_ratio=None, CNN_option=False):
    data_df = pd.read_csv(log_file)
    data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
    data_df['TimeSequence'] = data_df['TimeSequence'].map(literal_eval)
    data_df['PidSequence'] = data_df['PidSequence'].map(literal_eval)
    # Split train and test data
    return data_df, _split_idxs(data_df['Label'].values, train_ratio=train_ratio, test_ratio=test_ratio, CNN_option=CNN_option)


def csv_extracting(log_file, col_header_file, label_file):
    '''
    :param log_file: txt/log
    :param col_header_file: txt
    :param label_file: csv
    :return: None
    '''
    print("== STEP 1 ==")
    assert log_file.endswith('.txt') or log_file.endswith('.log')
    _load_HDFS(log_file, col_header_file, label_file)
    csv_file = log_file[: -4] + '.csv'
    _load_HDFS(csv_file, label_file=label_file, save_csv=True)
