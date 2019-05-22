from contextlib import contextmanager

import numpy as np
import six

import tfsnippet as spt
from tfsnippet.utils import get_default_session_or_error
from tfsnippet.scaffold import EventKeys
from tfsnippet.trainer import resolve_feed_dict, merge_feed_dict


class LogBlockEvaluator(spt.Evaluator):
    '''
    Evaluator designed for log blocks.
    When evaluating, it will not only record metrics, but also keep ll(log likelihood) for
    every line of input.
    '''
    def __init__(self, ll, *args, **kwargs):
        '''
        Construct a new :class:`LogBlockEvaluator`.

        Args:
            ll: Log likelihood tensor.
        '''
        self._ll = ll
        self._ll_array = None
        self._boundary = None
        super(LogBlockEvaluator, self).__init__(*args, **kwargs)

    def _run_batch(self, session, feed_dict):
        return session.run(list(six.itervalues(self.metrics)) + [self._ll],
                           feed_dict=feed_dict)

    @property
    def ll_array(self):
        '''
        Get the array of all log likelihoods.

        Returns:
            A array of all log likelihoods.
        '''
        return self._ll_array

    @property
    def boundary(self):
        '''
        Get the boundary.

        Returns:
            A float, boundary.
        '''
        return self._boundary

    def set_boundary(self, boundary):
        '''
        Args:
            boundary: The new boundary.
        '''
        self._boundary = boundary

    def run(self, feed_dict=None):
        """
        Run evaluation.

        Args:
            feed_dict: The extra feed dict to be merged with the already
                configured dict.  (default :obj:`None`)
        """
        @contextmanager
        def timeit():
            if self.time_metric_name is not None:
                with self.loop.timeit(self.time_metric_name):
                    yield
            else:
                yield

        session = get_default_session_or_error()
        metric_tensors = list(six.itervalues(self.metrics))
        metric_names = list(six.iterkeys(self.metrics))
        metric_values = []
        metric_weights = []

        _ll_list = []

        with timeit():
            # trigger before evaluation event
            self.events.fire(EventKeys.BEFORE_EXECUTION, self)

            for batch_data in self.data_flow:
                # prepare for the batch feed dict
                feed_dict = resolve_feed_dict(
                    merge_feed_dict(
                        self.feed_dict,
                        feed_dict,
                        zip(self.inputs, batch_data)
                    )
                )

                # inspect the batch weight
                if self._batch_weight_func is not None:
                    batch_weight = self._batch_weight_func(*batch_data)
                else:
                    batch_weight = 1.
                metric_weights.append(batch_weight)

                # run the mini-batch
                batch_values = self._run_batch(session, feed_dict)
                for i, v in enumerate(batch_values[: -1]):
                    if len(np.asarray(v).shape) != 0:  # pragma: no cover
                        raise ValueError(
                            'Metric is not a scalar: tensor {!r}, value {!r}.'.
                            format(v, metric_tensors[i])
                        )

                # accumulate the metrics
                metric_values.append(np.asarray(batch_values[: -1]))
                _ll_list.append(batch_values[-1])

            # now merge all batch metrics and do logging
            if metric_values:
                metric_values = np.average(
                    np.stack(metric_values, axis=0),
                    axis=0,
                    weights=np.asarray(metric_weights),
                )
                assert(len(metric_names) == len(metric_values))
                self._last_metrics_dict = metrics_dict = {
                    k: v for k, v in zip(metric_names, metric_values)
                }
                self.loop.collect_metrics(metrics_dict)

            # get ll array
            self._ll_array = np.concatenate(tuple(_ll_list))

            # trigger after evaluation event
            self.events.reverse_fire(EventKeys.AFTER_EXECUTION, self)
