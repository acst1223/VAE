import numpy as np
from tfsnippet.dataflows import DataFlow


def one_hot_flow(x, data_loader, batch_size, shuffle=False, skip_incomplete=False, random_state=None):
    '''
    Generate an one-hot-flow according to the given DataLoader object.
    '''
    x = np.asarray(x)

    # compose the data flow
    source_flow = DataFlow.arrays([x], batch_size, shuffle, skip_incomplete,
                                  random_state)
    map_flow = source_flow.map(lambda t: [data_loader.vectors[t]])
    return map_flow
