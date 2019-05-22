from pipeline.vae_dataloader import DataLoader
import os
from tfsnippet import DataFlow
import numpy as np


dataloader = DataLoader(os.path.join('D:/NetMan/codes/VAE',
                                     'pipeline',
                                     'HDFS',
                                     'data_instances_1e5.csv'),
                        0.3, 0.6, 10)
(x_train, y_train), (x_test, y_test), (x_validate, y_validate) = \
    (dataloader.x_train, dataloader.y_train), \
    (dataloader.x_test, dataloader.y_test), \
    (dataloader.x_validate, dataloader.y_validate)
x_train = np.asarray(x_train)
train_flow = DataFlow.arrays(
    [x_train], 25, shuffle=True, skip_incomplete=True)
map_flow = train_flow.map(lambda x: [np.sum(x, axis=1)])
i = 0
for [x] in map_flow:
    i += 1

    if i == 3:
        break
