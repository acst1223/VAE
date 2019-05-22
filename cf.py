import tensorflow as tf
import os
import sys

path = ''
if os.path.exists('./pipeline'):
    print('Config: PyCharm mode')
    path = './'
else:
    print('Config: Docker mode')
    path = '/root/'
HDFS_data = path + 'pipeline/HDFS/HDFS_1e5'
HDFS_result_png_prefix = path + 'pipeline/HDFS/result'
log_path = ''


def init(model_name):
    global log_path
    log_path = path + 'log/' + model_name + '.log'

def log(str):
    print(str)
    f = open(log_path, 'a')
    f.write('%s\n' % str)
    f.close()

def get_stat_path(data_path):
    return data_path[: -4] + '.stat.pkl'

def get_HDFS_result_png_name(template):
    return HDFS_result_png_prefix + '_' + template + '.png'
