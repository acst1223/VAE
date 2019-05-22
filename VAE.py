# -*- coding: utf-8 -*-
import functools
import os
import sys
sys.path.append(os.path.dirname(sys.modules['__main__'].__file__))
from argparse import ArgumentParser

import tensorflow as tf
from pprint import pformat
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from tfsnippet.examples.utils import (MLResults,
                                      print_with_title)

import numpy as np

from pipeline.vae_dataloader import DataLoader
from utils.log_block_evaluator import LogBlockEvaluator
from utils.decide_boundary import decide_boundary
from utils.log_file import log_file
from utils.one_hot_flow import one_hot_flow


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 20
    h = 10
    tot_templates = 29 # TODO: should not be hard-encoded
    x_dim = h * tot_templates

    # training parameters
    result_dir = None
    write_summary = False
    max_epoch = 15
    max_step = None
    batch_size = 15
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 500
    test_batch_size = 128

    # paths
    checkpoint_dir = os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                                  'train_z_20')
    train_log_file = os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                                  'log', 'log_z_20.log')


config = ExpConfig()


@spt.global_reuse
@add_arg_scope
def q_net(x, observed=None, n_z=None, is_initializing=False):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = functools.partial(
        spt.layers.act_norm, initializing=is_initializing)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=True,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.dense(h_x, 500)
        h_x = spt.layers.dense(h_x, 500)

    # sample z ~ q(z|x)
    z_mean = spt.layers.dense(h_x, config.z_dim, name='z_mean')
    z_logstd = spt.layers.dense(h_x, config.z_dim, name='z_logstd')
    z = net.add('z', spt.Normal(mean=z_mean, logstd=z_logstd), n_samples=n_z,
                group_ndims=1)

    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None, is_initializing=False):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = functools.partial(
        spt.layers.act_norm, initializing=is_initializing)

    # sample z ~ p(z)
    z = net.add('z', spt.Normal(mean=tf.zeros([1, config.z_dim]),
                                logstd=tf.zeros([1, config.z_dim])),
                group_ndims=1, n_samples=n_z)

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=True,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = z
        h_z = spt.layers.dense(h_z, 500)
        h_z = spt.layers.dense(h_z, 500)

    # sample x ~ p(x|z)
    x_logits = spt.layers.dense(h_z, config.x_dim, name='x_logits')
    x = net.add('x', spt.Bernoulli(logits=x_logits), group_ndims=1)

    return net


def get_normal_anomaly_ll(ll, y, x_count):
    assert np.sum(np.array(x_count)) == len(ll)
    normal_ll = []
    anomaly_ll = []
    idx = 0
    for i, c in enumerate(x_count):
        if y[i] == 0:
            normal_ll.append(np.min(ll[idx: idx + c]))
        else:
            anomaly_ll.append(np.min(ll[idx: idx + c]))
        idx += c
    return normal_ll, anomaly_ll


def stat_normal_anomaly_ll(evaluator, y, x_count, results):
    ll = evaluator.ll_array
    normal_ll, anomaly_ll = get_normal_anomaly_ll(ll, y, x_count)
    boundary, precision, recall, f1_score = decide_boundary(normal_ll, anomaly_ll)
    evaluator.set_boundary(boundary)
    results.update_metrics({'boundary': boundary})
    print('After epoch %d: Boundary: %g, Precision: %g, Recall: %g, F1-Score: %g' % (evaluator.loop.epoch,
                                                                                     boundary,
                                                                                     precision,
                                                                                     recall,
                                                                                     f1_score))


def get_result_during_test(evaluator, y, x_count, results):
    ll = evaluator.ll_array
    normal_ll, anomaly_ll = get_normal_anomaly_ll(ll, y, x_count)
    boundary = evaluator.boundary
    tp, fp = 0, 0
    for item in normal_ll:
        if item <= boundary:
            fp += 1
    for item in anomaly_ll:
        if item <= boundary:
            tp += 1
    precision = tp / (tp + fp)
    recall = tp / len(anomaly_ll)
    f1_score = 2 * precision * recall / (precision + recall)
    print('Test: Precision: %g, Recall: %g, F1-Score: %g' % (precision, recall, f1_score))
    results.update_metrics({'precision': precision, 'recall': recall, 'f1_score': f1_score})


@log_file(config.train_log_file)
def main():
    # parse the arguments
    arg_parser = ArgumentParser()
    spt.register_config_arguments(config, arg_parser, title='Model options')
    spt.register_config_arguments(spt.settings, arg_parser, prefix='tfsnippet',
                                  title='TFSnippet options')
    arg_parser.parse_args(sys.argv[1:])

    # print the config
    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs('plotting', exist_ok=True)
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None, config.x_dim), name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the output for initialization
    with tf.name_scope('initialization'), \
            spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
        init_q_net = q_net(input_x, is_initializing=True)
        init_chain = init_q_net.chain(
            p_net, observed={'x': input_x}, is_initializing=True)
        init_lb = tf.reduce_mean(init_chain.vi.lower_bound.elbo())

    # derive the loss and lower-bound for training
    with tf.name_scope('training'):
        train_q_net = q_net(input_x)
        train_chain = train_q_net.chain(p_net, observed={'x': input_x})
        vae_loss = tf.reduce_mean(train_chain.vi.training.sgvb())
        loss = vae_loss + tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_z)
        test_chain = test_q_net.chain(
            p_net, latent_axis=0, observed={'x': input_x})
        test_nll = -tf.reduce_mean(test_chain.vi.evaluation.is_loglikelihood())
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())
        test_ll = test_chain.vi.evaluation.is_loglikelihood()

    # derive the optimizer
    with tf.name_scope('optimizing'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=params)
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(grads)

    # prepare for training and testing data
    dataloader = DataLoader(os.path.join(os.path.dirname(sys.modules['__main__'].__file__),
                                         'pipeline',
                                         'HDFS',
                                         'data_instances_full.csv'),
                            0.3, 0.6, 10)
    (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = \
        (dataloader.x_train, dataloader.y_train), \
        (dataloader.x_test, dataloader.y_test), \
        (dataloader.x_validate, dataloader.y_validate)

    x_test_count, x_validate_count = dataloader.x_test_count, dataloader.x_validate_count

    train_flow = one_hot_flow(
        x_train, dataloader, config.batch_size, shuffle=True, skip_incomplete=True)
    validate_flow = one_hot_flow(
        x_validate, dataloader, config.test_batch_size)
    test_flow = one_hot_flow(
        x_test, dataloader, config.test_batch_size)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        # initialize the network
        for [x] in train_flow:
            print('Network initialized, first-batch loss is {:.6g}.\n'.
                  format(session.run(init_lb, feed_dict={input_x: x})))
            break

        # train the network
        with spt.TrainLoop(params,
                           var_groups=['q_net', 'p_net'],
                           max_epoch=config.max_epoch,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           checkpoint_dir=config.checkpoint_dir,
                           checkpoint_epoch_freq=1,
                           checkpoint_max_to_keep=10) as loop:
            trainer = spt.Trainer(
                loop, train_op, [input_x], train_flow,
                metrics={'loss': loss},
                summaries=tf.summary.merge_all(spt.GraphKeys.AUTO_HISTOGRAM)
            )
            trainer.anneal_after(
                learning_rate,
                epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            validate_evaluator = LogBlockEvaluator(
                test_ll,
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb},
                inputs=[input_x],
                data_flow=validate_flow,
                time_metric_name='validate_time'
            )
            validate_evaluator.events.on(
                spt.EventKeys.AFTER_EXECUTION,
                lambda e: results.update_metrics(validate_evaluator.last_metrics_dict)
            )
            validate_evaluator.events.on(
                spt.EventKeys.AFTER_EXECUTION,
                lambda e: stat_normal_anomaly_ll(validate_evaluator, y_validate, x_validate_count, results)
            )
            trainer.evaluate_after_epochs(validate_evaluator, freq=1)
            trainer.log_after_epochs(freq=1)
            trainer.run()

        # If the model has not been trained, then it should
        # get an extra validation to get boundary.
        if 'boundary' not in results.metrics_dict:
            with spt.TrainLoop(params,
                               var_groups=['q_net', 'p_net'],
                               max_epoch=1) as loop:
                trainer = spt.Trainer(
                    loop, tf.exp(tf.constant(0.0)), [input_x], train_flow,
                    metrics={}
                )
                validate_evaluator = LogBlockEvaluator(
                    test_ll,
                    loop,
                    metrics={'test_nll': test_nll, 'test_lb': test_lb},
                    inputs=[input_x],
                    data_flow=validate_flow,
                    time_metric_name='validate_time'
                )
                validate_evaluator.events.on(
                    spt.EventKeys.AFTER_EXECUTION,
                    lambda e: results.update_metrics(validate_evaluator.last_metrics_dict)
                )
                validate_evaluator.events.on(
                    spt.EventKeys.AFTER_EXECUTION,
                    lambda e: stat_normal_anomaly_ll(validate_evaluator, y_validate, x_validate_count, results)
                )
                trainer.evaluate_after_epochs(validate_evaluator, freq=1)
                trainer.run()

        # testing
        with spt.TrainLoop(params,
                           var_groups=['q_net', 'p_net'],
                           max_epoch=1) as loop:
            trainer = spt.Trainer(
                loop, tf.exp(tf.constant(0.0)), [input_x], train_flow,
                metrics={}
            )
            test_evaluator = LogBlockEvaluator(
                test_ll,
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb},
                inputs=[input_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )
            test_evaluator.set_boundary(results.metrics_dict['boundary'])
            test_evaluator.events.on(
                spt.EventKeys.AFTER_EXECUTION,
                lambda e: get_result_during_test(test_evaluator, y_test, x_test_count, results)
            )
            trainer.evaluate_after_epochs(test_evaluator, freq=1)
            trainer.run()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
