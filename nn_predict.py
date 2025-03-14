from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
import tensorflow as tf
set_random_seed(2)

import argparse
import ast
import datetime as dt
from functools import partial
import logging
from collections import namedtuple
import os
from pathlib import Path, PurePath
import random
import time
from time import sleep

import keras
from keras.backend import squeeze
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import model_from_json, Sequential
from keras.utils import multi_gpu_model

import numpy as np
from numpy import array
import pandas as pd
import requests
from access import get_xyt_arrs
import accesscloud
from accesscloud import download_s3_file

# Custom_recurrents
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec

FINISHED_DEV_ACTIVITY = 'fit and profiling'

TRAIN_LOG = './caar.log'
TRAIN_LOG = './test.log'
logger = logging.getLogger('nn_predict_app')
hdlr = logging.FileHandler(TRAIN_LOG)
formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(funcName)20s() %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

API_DOMAIN = 'https://api.paperspace.io'
LOG_DOMAIN = 'https://logs.paperspace.io'



Jobstats = namedtuple('Jobstats', ['jobid', 'machine_type', 'container',
                                       'exit_code', 'usage_rate', 'cpu_count',
                                       'cpu_mem', 'cpu_model'])


#  https://github.com/keunwoochoi/keras_callbacks_example/my_callbacks.py

class AccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []
        self.val_accuracy = []
        self.epoch_accuracy = []


    def on_batch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('acc'))
        #self.batch_accuracy.append(logs.get('val_acc'))


    def on_epoch_end(self, epoch, logs={}):
        # self.losses.append(logs.get('loss'))
        self.epoch_accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))


PATIENCE = 20
MIN_DELTA = 0.001

earlystop = EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA,
                          patience=PATIENCE, verbose=2, mode='auto')

def predict_and_store_results(ids, data_dir=None, data='on_off', suffix=None,
                              weights_suffix=None, len_in=None, len_out=None,
                              model_dir=None, write_examples=False,
                              overwrite=False, predict=True, val_split=None,
                              plot_sid=False, epochs=None, patience=None,
                              optimizer=None, ext=None):
    input_metas = {}
    results = {}
    for sid in ids:
        output = train_for_device(sid, data_dir=data_dir, data=data,
                                  suffix=suffix, len_in=len_in,
                                  len_out=len_out,
                                  write_examples=write_examples,
                                  overwrite=overwrite, predict=predict,
                                  val_split=val_split, epochs=epochs,
                                  patience=patience, optimizer=optimizer)
        metrics, input_meta = output
        if metrics:
            sid_res = summarize_device_training(sid, metrics, model_dir,
                                                weights_suffix, ext, len_in,
                                                len_out, epochs,
                                                plot_sid=plot_sid)
            results[sid] = sid_res
        input_metas[sid] = input_meta

    return results, input_metas


def remaining_to_train(headers, package):
    data, mod_dev_keys, metadata = get_data_set(package)
    dev_keys = [k[1:] for k in mod_dev_keys]
    trained = ids_trained_prev_jobs(headers)
    trained_ids = trained.keys()
    remaining = set(dev_keys) - set(trained_ids)
    return len(remaining)


def install_data(pkg):
    # force to avoid y/n prompt; does not re-download
    quilt.install(pkg, force=True)


def get_data_set(pkg):
    install_data(pkg)
    # from quilt.data.nickpowersys import constant
    constant = quilt.load(pkg)
    g_keys = constant._group_keys()
    dev_keys = [k for k in g_keys if k != 'metadata']

    return constant, dev_keys, constant['metadata']


def profile_ex_creation(log_file, input_metas, begin_str, end_str,
                        val_split=None):

    print('before time_deltas_from_log')
    dev_ids_begin, deltas = time_deltas_from_log(log_file, begin_str, end_str)
    print('after time_deltas_from_log')

    creation_times = []
    creation_times_per_day = []
    total_exs = []
    print('input_metas: {}'.format(input_metas))
    print('dev_ids begin: {}'.format(dev_ids_begin))
    for d, dev_id in enumerate(dev_ids_begin):
        logger.info('dev_id %s', dev_id)
        print('devid {}'.format(dev_id))
        v = input_metas[dev_id]
        print('v {}'.format(v))
        ex_per_day = v['ex_per_day']
        ex_days = len(v['ex_dates'])
        total_ex = round(ex_days * ex_per_day * 1.0/(1 - val_split))
        total_exs.append(total_ex)

        ex_creation_time = deltas[d]
        creation_times.append(ex_creation_time)
        creation_times_per_day.append(ex_creation_time/ex_days)
        assert len(creation_times) > 0
        assert len(creation_times_per_day) > 0
        assert len(total_exs) > 0

    ex_creation_profiled = {'creation_time_per_day': creation_times_per_day,
                            'ex_creation_time': creation_times,
                            'total_exs': total_exs,
                            'dev_id': dev_ids_begin}
    profile_df = pd.DataFrame.from_dict(ex_creation_profiled)
    profile_df.set_index('dev_id', inplace=True)
    print('len of df: {}'.format(len(profile_df)))
    return profile_df


def profile_training_for_id(dev_id=None, input_meta=None, begin=None,
                            end=None, stopped_epoch=None, max_epochs=None,
                            training_loss=None, val_loss=None, val_split=None,
                            model=None):
    training_time = end - begin
    ex_per_day = input_meta['ex_per_day']
    ex_days = len(input_meta['ex_dates'])
    total_exs = round(ex_days * ex_per_day * 1.0 / (1 - val_split))

    time_per_day = training_time / ex_days
    time_per_ex = training_time / total_exs
    train_keys = ['training_time_per_day', 'training_time_per_ex',
                  'training_time', 'total_exs', 'total_days',
                  'stopped_epoch', 'max_epochs', 'training_loss',
                  'val_loss', 'dev_id', 'model']

    train_vals = [time_per_day, time_per_ex, training_time, total_exs,
                  ex_days, stopped_epoch, max_epochs, training_loss,
                  list(val_loss), dev_id, model]
    train_profiled = dict([(k, v) for k, v in zip(train_keys, train_vals)])

    logger.info('training time dict for id {}:{}'.format(dev_id,
                                                         str(train_profiled)))
    return train_profiled


def profile_training_local(log_file, input_metas, begin_str, end_str,
                           val_split=None):

    dev_ids_begin, deltas = time_deltas_from_log(log_file, begin_str, end_str)

    training_times = []
    training_times_per_day = []
    training_times_per_ex = []
    total_exs = []
    days = []
    for d, dev_id in enumerate(dev_ids_begin):
        logger.info('dev_id %s', dev_id)
        v = input_metas[dev_id]
        ex_per_day = v['ex_per_day']
        ex_days = len(v['ex_dates'])
        total_ex = round(ex_days * ex_per_day * 1.0 / (1 - val_split))
        total_exs.append(total_ex)

        training_time = deltas[d]
        training_times.append(training_time)
        training_times_per_day.append(training_time / ex_days)
        training_times_per_ex.append(training_time / total_ex)
        days.append(ex_days)
    train_profiled = {'training_time_per_day': training_times_per_day,
                      'training_time_per_ex': training_times_per_ex,
                      'training_time': training_times,
                      'total_exs': total_exs,
                      'total_days': days,
                      'dev_id': dev_ids_begin}
    profile_df = pd.DataFrame.from_dict(train_profiled)
    profile_df.set_index('dev_id', inplace=True)
    return profile_df


def profile_training_paperspace(jobid, headers, projecct='tlproject',
                                train_func='LSTM_model', log_desc='lstm',
                                user='nickpowersys', datapkg='constant'):
    machine_stats = job_machine_stats(jobid, headers)

    dev_ids, deltas = training_times_from_logs(jobid, headers,
                                               train_func=train_func,
                                               log_desc=log_desc)

    package = '/'.join([user, datapkg])
    pkg = quilt.load(package)
    metadata = pkg['metadata']

    stop_epoch_dict, max_epochs = training_epochs_from_logs(jobid, headers)

    training_times = []
    training_times_per_day = []
    training_times_per_ex = []
    training_times_per_epoch = []
    training_exs = []
    days = []
    stop_epochs = []

    for dev, training_time in zip(dev_ids, deltas):
        training_times
        training_times.append(training_time)
        meta_node = metadata['n' + str(dev)]

        ex_days = np.size(meta_node['ex_dates']())
        training_times_per_day.append(training_time / ex_days)
        days.append(ex_days)

        train_exs = meta_node['num_train']().item()
        training_exs.append(train_exs)
        training_times_per_ex.append(training_time / train_exs)

        stop_epoch = stop_epoch_dict[dev]
        training_times_per_epoch.append(training_time / stop_epoch)
        stop_epochs.append(stop_epoch)

    train_profiled = {'training_time_per_day': training_times_per_day,
                      'training_time_per_ex': training_times_per_ex,
                      'training_time_per_epoch': training_times_per_epoch,
                      'training_time': training_times,
                      'total_exs': training_exs,
                      'total_days': days,
                      'stop_epoch': stop_epochs,
                      'max_epochs': [max_epochs] * len(dev_ids),
                      'dev_id': dev_ids}
    profile_df = pd.DataFrame.from_dict(train_profiled)
    profile_df.set_index('dev_id', inplace=True)
    return profile_df


def job_machine_stats(jobid, headers, project='tlproject'):
    payload = {'project': project}
    list_job = '/'.join([API_DOMAIN, 'jobs', 'getJobs'])
    r = requests.get(list_job, params=payload, headers=headers)
    job_stats = r.json()

    for job in job_stats:
        if job['id'] == jobid:
            exit_code = job['exitCode']
            container = job['container']
            machine_type = job['machineType']
            usage = job['usageRate']
            cpu = job['cpuModel']
            cpu_count = job['cpuCount']
            cpu_mem = job['cpuMem']

    stats = Jobstats(jobid=jobid, machine_type=machine_type,
                     exit_code=exit_code, container=container,
                     usage_rate=usage, cpu_count=cpu_count, cpu_mem=cpu_mem,
                     cpu_model=cpu)

    return stats


def training_times_from_logs(jobid, headers, train_func='LSTM_model',
                             log_desc='lstm'):
    payload = {'jobId': jobid}
    log_call = '/'.join([LOG_DOMAIN, 'jobs', 'logs'])
    r = requests.get(log_call, params=payload, headers=headers)
    log_content = r.json()

    begin_str = ' '.join([train_func + '()', 'begin', log_desc])  # 'LSTM_model() begin lstm'
    end_str = ' '.join([train_func + '()', 'end', log_desc])  # 'LSTM_model() end lstm'

    log_times_msgs = [(pd.Timestamp(l['timestamp']), l['message'])
                      for l in log_content]

    lstm_begins = [t_msg[0] for t_msg in log_times_msgs if begin_str in t_msg[1]]
    lstm_ends = [t_msg[0] for t_msg in log_times_msgs if end_str in t_msg[1]]
    dev_ids = [logged[1].split()[-1] for logged in log_times_msgs
               if begin_str in logged[1]]
    training_times = [e - b for b, e in zip(lstm_begins, lstm_ends)]
    return dev_ids, training_times


def training_epochs_from_logs(jobid, headers,
                              train_func='summarize_device_training',
                              stop_desc='stopped_epoch',
                              log_desc2='max_epochs'):
    payload = {'jobId': jobid}
    log_call = '/'.join([LOG_DOMAIN, 'jobs', 'logs'])
    r = requests.get(log_call, params=payload, headers=headers)
    log_content = r.json()

    epochs_str = ' '.join([train_func + '()', "{'" + stop_desc])

    log_times_msgs = [l['message'] for l in log_content
                      if epochs_str in l['message']]
    epochs_msgs = (msg[msg.find('{'): msg.find('}') + 1]
                   for msg in log_times_msgs if msg.find('{') >= 0)
    info_dicts = (ast.literal_eval(msg) for msg in epochs_msgs)
    ids_stops = dict([(info['id'], info['stopped_epoch'] + 1)
                     for info in info_dicts])
    max_epoch = None
    for msg in log_times_msgs:
        if 'max_epochs' in msg:
            max_epoch = int(msg[msg.find('max_epochs') + 12: msg.find(", 'id")])
            break

    return ids_stops, max_epoch


def time_deltas_from_log(log_file, begin_str, end_str):
    begins = []
    ends = []
    dev_ids_begin = []
    dev_ids_end = []
    prev_id_begin = None
    prev_id_end = None
    print('begin_str: {}'.format(begin_str))

    with open(log_file) as f:
        for logged in f:
            print('logged: {}'.format(logged))
            #dev_id = int(logged.split()[-1])
            dev_id = logged.split()[-1]
            print('dev_id: {}'.format(dev_id))
            begin_str_in_logged = begin_str in logged
            dev_id_in_logged = dev_id in logged
            print('begin_str in logged: {}'.format(begin_str_in_logged))
            print('dev_id in logged: {}'.format(dev_id_in_logged))
            if all((begin_str in logged, dev_id in logged,
                    dev_id != prev_id_begin)):
                raw_datetime = logged.split()[0:2]
                begins.append(raw_datetime)

                dev_ids_begin.append(dev_id)
                prev_id_begin = dev_id

            elif all((end_str in logged, dev_id in logged,
                     dev_id != prev_id_end)):
                end_str_in_logged = end_str in logged
                print('end_str in logged: {}'.format(end_str_in_logged))
                raw_datetime = logged.split()[0:2]
                ends.append(raw_datetime)

                dev_ids_end.append(dev_id)
                prev_id_end = dev_id
                # logger.info('prev_id_begin %s', dev_id)

        assert dev_ids_begin == dev_ids_end

        deltas = []

        for b, e in zip(begins, ends):
            begin_t, end_t = (pd.Timestamp(' '.join(t)) for t in (b, e))
            assert end_t > begin_t
            deltas.append(end_t - begin_t)

        print('dev_ids_begin: {}'.format(dev_ids_begin))
        print('deltas: {}'.format(deltas))

        return dev_ids_begin, deltas


def train(ids=None, data=None, val_split=None, patience=None,
          batch_size=None, epochs=None, weights_suffix=None, weights_dir=None,
          optimizer=None, ext=None, checkpoint_callbacks=False, completed=None,
          model=None, sep=','):

    if isinstance(model, tuple):
        print('len of model: ', len(model))

    len_in, len_out = len_of_x_and_y(data)

    previous_jobs_exist = True if completed else False

    input_metas = {}
    results = {}

    if ids is None:
        print('ids is none. Basing ids off data.')
        dev_ids = list(data.keys())
    else:
        print('ids is not None. Basing ids off parameter.')
        dev_ids = ids.split(sep)
    print('in train. ids: {}'.format(', '.join(dev_ids)))

    if previous_jobs_exist:
        print('completed: {}'.format(completed))
        devs_jobs = dict(completed)
        for dev_id, jobid in devs_jobs.items():
            logging.info("dev_trained:job_id{'%s': '%s'}" %
                         (dev_id, jobid))

        devs_trained = devs_jobs.keys()
        to_complete = list(set(dev_ids) - set(devs_trained))
    else:
        to_complete = dev_ids
    print('to complete: {}'.format(to_complete))

    print('to complete: {}'.format(to_complete))


    model_label, model_func = model # 'LSTM', actual model function

    print('Model: ', model_label)


    for dev in to_complete:
        begin = time.time()
        #sidnode = data[mod_id]
        #X, Y, T = sidnode.X(), sidnode.Y(), sidnode.T()
        dev_data = data[dev]
        X, Y, T = dev_data['x'], dev_data['y'], dev_data['t']

        print(f'Shape of X: {X.shape}')
        print(f'Shape of Y: {Y.shape}')
        print(f'Shape of T: {T.shape}')

        metrics = model_func(X, Y, len_in, len_out, val_split=val_split,
                             patience=patience, dev_id=dev,
                             weights_suffix=weights_suffix,
                             weights_dir=weights_dir, batch_size=batch_size,
                             epochs=epochs, optimizer=optimizer,
                             checkpoint_callbacks=checkpoint_callbacks)
        results[dev] = summarize_device_training(int(dev), metrics,
                                                 weights_dir,
                                                 weights_suffix, ext,
                                                 len_in, len_out, epochs)
        end = time.time()

        input_metas[dev] = summarize_input(dev, X, T, len_in, val_split)

        epoch_metrics = epochs_val_loss_from_metrics(metrics)
        stopped, max_epochs, training_loss, val_loss = epoch_metrics

        profile_training_for_id(dev_id=dev, input_meta=input_metas[dev],
                                begin=begin, end=end, stopped_epoch=stopped,
                                max_epochs=max_epochs,
                                training_loss=training_loss,
                                val_loss=val_loss, val_split=val_split,
                                model=model_label)

        logger.info(model_fit_profiling_str(model_label, dev))

    return results, input_metas


def model_fit_profiling_str(model_str, dev):
    return 'end {} {} for dev {}'.format(model_str, FINISHED_DEV_ACTIVITY,
                                         dev)


def epochs_val_loss_from_metrics(metrics):
    (_, _, _, _, _, _, _, _,
     _, _, _, _, val_loss, _, _,
     _, stopped_epoch, _, max_epochs) = metrics
    training_epochs_patience = min(len(val_loss), max_epochs)
    training_loss = np.mean(val_loss[-training_epochs_patience:])
    return stopped_epoch + 1, max_epochs, training_loss, val_loss


def add_prefix_for_quilt(ids):
    return ['n' + id for id in ids]


def remove_prefix_for_quilt_data(ids):
    return [id[1:] for id in ids]


def len_of_x_and_y(data):
    # sample_sid = dev_keys[0]
    random_id = random.choice(list(data.keys()))
    # sample_data = data[sample_sid]
    random_X, random_Y = data[random_id]['x'], data[random_id]['y']
    #random_X, random_Y = sample_data.X(), sample_data.Y()
    len_in, len_out = random_X.shape[1], random_Y.shape[1]
    return len_in, len_out


def train_and_summarize(ids, data_dir=None, data='on_off', suffix=None,
                        weights_suffix=None, len_in=None, len_out=None,
                        model_dir=None, write_examples=False,
                        overwrite=False, predict=True, val_split=None,
                        plot_sid=False, epochs=None, patience=None,
                        weights_dir=None, summary_ext=None, quilt_pkg=None):
    input_metas = {}
    results = {}
    for sid in ids:
        # train_for_device
        output = train_for_device(sid, data_dir=data_dir, data=data,
                                  suffix=suffix, weights_suffix=weights_suffix,
                                  len_in=len_in, len_out=len_out,
                                  write_examples=write_examples,
                                  overwrite=overwrite, predict=predict,
                                  val_split=val_split, epochs=epochs,
                                  patience=patience, weights_dir=weights_dir)
        metrics, input_meta = output

        if metrics:
            sid_res = summarize_device_training(sid, metrics, model_dir,
                                                weights_suffix, summary_ext,
                                                len_in, len_out, epochs,
                                                plot_sid=plot_sid)
            results[sid] = sid_res
        input_metas[sid] = input_meta

    return results, input_metas


def summarize_device_training(sid, metrics, model_dir, weights_suffix, ext,
                              len_in, len_out, epochs, plot_sid=False):
    # metrics, earliest, latest, latest_train, examples_per_day, num_ex = output
    (model, metrics_names, val, hist1, accs, hist2, val_accs, hist3,
     epoch_accs, hist4, loss, hist5, val_loss, hist6, patience,
     hist7, stopped_epoch, hist8, max_epochs) = metrics

     # can't have metrics twice

    pct = lambda x: round(x*100, 2)
    accs_rnd = array([pct(acc) for acc in accs])
    val_accs_rnd = array([pct(vacc) for vacc in val_accs])
    # epoch_accs_rnd = array([pct(eacc) for eacc in epoch_accs])
    epoch_accs_rnd = None
    loss_arr, val_loss_arr = array(loss), array(val_loss)
    val_round = pct(val[1])

    avg_val_loss = np.mean(val_loss_arr[-patience:])

    #results = dict([('earliest', earliest), ('latest', latest),
    #                ('num_ex', num_ex), ('exs_per_day', examples_per_day),
    results = dict([(metrics_names[1], val_round), (hist1, accs_rnd),
                    (hist2, val_accs_rnd), (hist3, epoch_accs_rnd),
                    (hist4, loss_arr), (hist5, val_loss_arr),
                    (hist6, patience), ('avg_val_loss', avg_val_loss),
                    (hist7, array([stopped_epoch])),
                    (hist8, array([max_epochs]))])
    overall_acc = array([val_round])

    #logger.info('stopped_epoch %s, max_epochs %s id %s' %
    #            (stopped_epoch, max_epochs, sid))
    #logger.info("{'stopped_epoch':%s, 'max_epochs':%s, 'id':'%s'}" %
    #            (stopped_epoch, max_epochs, sid))
    # logger.info('stopped_epoch {}, max_epochs {}, id {}'.
    #             format(stopped_epoch, max_epochs, sid))
    # epochs_keys = ['stopped_epoch', 'max_epochs', 'id']
    # epochs_vals = [stopped_epoch, max_epochs, sid]
    # summary = dict([(k, v) for k, v in zip(epochs_keys, epochs_vals)])
    # logger.info('training epochs dict for id {}:{}'.format(sid, str(summary)))

    res_path = training_results_file(sid, weights_suffix, model_dir, ext,
                                     len_in, len_out, epochs, patience)

    try:
        np.savez_compressed(res_path, overall_acc=overall_acc, acc=accs_rnd,
                            val_acc=val_accs_rnd, epoch_acc=epoch_accs_rnd,
                            loss=loss_arr, val_loss=val_loss_arr,
                            patience=array([patience]),
                            avg_val_loss=array([avg_val_loss]),
                            stopped_epoch = array([stopped_epoch]),
                            max_epochs=array([max_epochs]))
        logging.info('%s written.', res_path)
    except IOError:
        logging.info('Unable to write to %s', res_path)

    in_out = '_'.join(['in', str(len_in), 'out', str(len_out)])
    num_epochs_patience = '_'.join([str(epochs), 'epochs', str(patience),
                                    'patience'])
    weights_file = '_'.join(['model', 'weights', in_out, weights_suffix,
                             num_epochs_patience,
                             '{:03d}'.format(sid) + '.h5'])
    weights_path = os.path.join(model_dir, weights_file)
    if PurePath(weights_path).is_absolute() and Path(weights_path).parent.exists():
        model.save_weights(weights_path)
    else:
        print('the path as stated does not exist')

    logger.info('end summarizing fit for id %s', str(sid))

    if plot_sid:
        acc, val_acc = accs_rnd, val_accs_rnd
        plot_train_test(acc, val_acc)
        plot_loss_val_loss(loss_arr, val_loss_arr)

    return results


def review_training_performance(ids, weights_suffix, model_dir, ext, len_in,
                                len_out, epochs, patience, metric=None):
    paths = (training_results_file(sid, weights_suffix, model_dir, ext, len_in,
                                   len_out, epochs, patience)
             for sid in ids)
    perf = {}
    for sid_path in zip(ids, paths):
        sid, path = sid_path
        train_arrs = np.load(path)
        if metric is not None:
            perf[sid] = train_arrs[metric]
        else:
            perf[sid] = train_arrs
    return perf


def training_results_file(sid, weights_suffix, model_dir, ext, len_in,
                          len_out, epochs, patience):
    in_out = '_'.join(['in', str(len_in), 'out', str(len_out)])
    num_epochs_patience = '_'.join([str(epochs), 'epochs', str(patience),
                                    'patience'])

    res_file = '_'.join(['train', in_out, weights_suffix, num_epochs_patience,
                         str(sid)]) + ext
    return os.path.join(model_dir, res_file)


def plot_train_test(acc, val_acc):
    num_train, num_val = acc.shape[0], val_acc.shape[0]
    train_per_val = int(num_train/num_val)
    ax = figure().gca()
    ax.plot(acc[::train_per_val])
    ax.plot(val_acc)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    show()


def plot_loss_val_loss(loss, val_loss):
    num_train, num_val = loss.shape[0], val_loss.shape[0]
    train_per_val = int(num_train/num_val)
    ax = figure().gca()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log loss')
    ax.plot(loss[::train_per_val])
    ax.plot(val_loss)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    show()


def plot_multiple_dev_losses(losses):
    ax = figure().gca()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    rands = [random.randint(0, len(losses) - 1) for i in range(3)]
    logging.info('rands %s', rands)
    logging.info('len of losses %s', len(losses))
    logging.debug('type of losses %s', type(losses))
    for rand in rands:
        dev_losses = losses[rand]
        logging.debug('type of dev_losses %s', type(dev_losses))
        logging.debug('len of dev_losses %s', len(dev_losses))
        dev_id, train_losses, val_losses = dev_losses
        num_train, num_val = train_losses.shape[0], val_losses.shape[0]
        train_per_val = int(num_train / num_val)
        ax.plot(train_losses[::train_per_val], label='loss Dev' + dev_id)
        ax.plot(val_losses, label='val loss Dev' + dev_id)
        ax.legend()
    show()


def train_for_device(sid, data_dir=None, data='on_off', suffix=None,
                     weights_suffix=None, len_in=None, len_out=None,
                     write_examples=False, overwrite=False, predict=True,
                     val_split=None, epochs=None, patience=None,
                     weights_dir=None, optimizer='adam'):

    X, Y, T = single_device_examples(sid, data_dir=data_dir, suffix=suffix,
                                     data=data, len_in=len_in, len_out=len_out,
                                     write_examples=write_examples,
                                     overwrite=overwrite)
    if predict:
        metrics = LSTM_model(X, Y, len_in, len_out, val_split=val_split,
                             epochs=epochs, patience=patience, dev_id=sid,
                             weights_suffix=weights_suffix,
                             weights_dir=weights_dir, optimizer=optimizer)
    else:
        metrics = False

    input_meta = summarize_input(sid, X, T, len_in, val_split, data_dir,
                                 suffix)

    return metrics, input_meta


def summarize_input_for_all(user, package, data, dev_keys, val_split):
    len_in, len_out = len_of_x_and_y(data, dev_keys)
    label_prefix = '/'.join([user, package, 'metadata'])
    summarize_partial = partial(summarize_input, len_in=len_in,
                                val_split=val_split)
    for d, mod_id in enumerate(dev_keys):
        sidnode = data[mod_id]
        node_label = '/'.join([label_prefix, mod_id])
        x, t = sidnode.X(), sidnode.T()
        input_meta = summarize_partial(sid=mod_id, X=x, T=t)
        for k, v in input_meta.items():
            label = '/'.join([node_label, k])
            raw_type = type(v)
            if raw_type == dt.date:
                arr = array(np.datetime64(v).astype('datetime64[D]'))
            elif raw_type == set:
                arr = (pd.DatetimeIndex(list(v))
                       .values
                       .astype('datetime64[D]'))
            else:
                arr = array(v)
            quilt.build(label, arr)


def summarize_input(sid, X, T, len_in, val_split, data_dir=None, suffix=None,
                    write=False):
    if type(T) == np.ndarray:
        pd_index = pd.DatetimeIndex(T)
        ex_starts = pd_index[::len_in]
    elif type(T) == pd.DataFrame:
        ex_starts = T.index[::len_in]

    num_train = int(round(ex_starts.size * (1 - val_split)))

    train_starts = ex_starts[:num_train]

    train_days = set(d.date() for d in train_starts)

    num_train_days = len(train_days)

    ex_per_day = round(num_train / num_train_days)

    first = ex_starts[0].date()
    last = ex_starts[-1].date()
    last_train = train_starts[-1].date()

    ex_dates = set(d.date() for d in ex_starts)

    ex_date_arr = (pd.DatetimeIndex(list(ex_dates))
                   .values
                   .astype('datetime64[D]'))

    if write:
        input_file = '_'.join(['input', suffix, str(sid)])
        input_path = os.path.join(data_dir, input_file)
        try:
            np.savez_compressed(input_path, first=array([first]),
                                last=array([last]),
                                last_train=array([last_train]),
                                ex_per_day=array([ex_per_day]),
                                num_train=array([num_train]),
                                ex_dates=ex_date_arr)
            logging.info('%s written.', input_path)
        except IOError:
            logging.info('Unable to write to %s', input_path)

    input_meta = dict([('first', first), ('last', last),
                      ('last_train', last_train), ('ex_per_day', ex_per_day),
                      ('num_train', num_train), ('ex_dates', ex_dates)])
    return input_meta


def random_predictions_and_actuals(ids, data_dir=None, data='on_off',
                                   len_in=None, len_out=None, suffix=None,
                                   model_dir=None):
    rand_predicts = []
    actuals = []
    kwargs = dict([('data_dir', data_dir), ('data', data),
             ('len_in', len_in), ('len_out', len_out),
             ('suffix', suffix),('model_dir', model_dir)])
    for sid in ids:
        yhats, rand_actuals = predict_random_examples(sid, **kwargs)
        rand_predicts.append(yhats)
        actuals.append(rand_actuals)

    rand_yhats = np.stack(tuple(rand_predicts))
    for i in range(0,len_out, 5):
        file_name = 'rand_yhats' + '_' + str(i) + '.csv'
        rf = os.path.join(model_dir, file_name)
        rand_yhat = rand_yhats[:,:,i,:].flatten()
        np.savetxt(rf, rand_yhat)

    rand_ys = np.stack(tuple(actuals))
    for i in range(0, len_out, 5):
        file_name = 'rand_ys' + '_' + str(i) + '.csv'
        rf = os.path.join(model_dir, file_name)
        rand_y = rand_ys[:, i, :].flatten()
        np.savetxt(rf, rand_y)


def predict_random_aggregation(ids, data_dir=None, len_in=None,
                               len_out=None, suffix=None, model_dir=None,
                               references='ends', hour=None):
    predict_partial = partial(predict_random_examples, data_dir=data_dir,
                              len_in=len_in, len_out=len_out, suffix=suffix,
                              model_dir=model_dir, references=references,
                              hour=hour)
    preds = {}

    sid_mins_elapsed = []
    sid_preds = []
    sid_actuals = []

    for sid in ids:
        pred_for_hour = predict_partial(sid=sid)
        for pred in pred_for_hour:
            hour, minutes_elapsed, predicted_Y, val_Y = pred
            if minutes_elapsed > 0:
                ts_yhat = np.concatenate((np.full(minutes_elapsed, np.nan,
                                         dtype=float), predicted_Y))
                ts_y = np.concatenate((np.full(minutes_elapsed, np.nan,
                                      dtype=float), val_Y))
            else:
                ts_yhat, ts_y = predicted_Y, val_Y

            sid_mins_elapsed.append(minutes_elapsed)
            sid_preds.append(ts_yhat)
            sid_actuals.append(ts_y)

        preds[sid] = (array(sid_mins_elapsed), array(sid_preds),
                      array(sid_actuals))
        del sid_mins_elapsed[:]
        del sid_preds[:]
        del sid_actuals[:]


def yhat_y(summary, val_example_i, X, Y, model, len_in, len_out, max_exs):
    val_examples = summary[val_example_i:, 0]
    val_minutes = array(summary[val_example_i:, 1])

    preds = np.full((max_exs, len_out), np.nan)
    actuals = np.full(preds.shape, np.nan)

    if max_exs is not None and val_examples.size >= max_exs:
        logging.debug('type of val_examples: %s', type(val_examples[0]))
        full_exs = set(zip(val_examples, val_minutes))
        if val_examples.size > max_exs:
            while len(full_exs) > max_exs:
                rand_ex = random.choice(tuple(full_exs))
                full_exs.remove(rand_ex)
        selected_exs_mins = np.array(list(full_exs))
        selected_exs = selected_exs_mins[:, 0].astype(np.int64)
        selected_mins = selected_exs_mins[:, 1].astype(np.int64)

    elif max_exs is not None and val_examples.size < max_exs:
        if val_examples.size == 0:
            val_examples = summary[val_example_i-1:, 0]
            val_minutes = summary[val_example_i-1:, 1]
            num_exs = val_examples.size
            if num_exs == 0:
                raise ValueError('hey num_ex is 0')
        selected_exs = np.full(max_exs, np.nan, dtype=int)
        selected_exs[:val_examples.size] = array(val_examples)
        selected_mins = np.full(max_exs, np.nan, dtype=int)
        selected_mins[:val_minutes.size] = array(val_minutes)

        for ex in range(val_examples.size, max_exs):
            rand_ex = random.randint(0, val_examples.size - 1)
            selected_exs[ex] = val_examples[rand_ex]
            selected_mins[ex] = val_minutes[rand_ex]

    rows = np.arange(max_exs)
    np.random.shuffle(rows)
    for row_ex in zip(rows, selected_exs):
        row, ex = row_ex
        val_X = np.array(X[ex, :, :]).reshape((1, len_in, 1))
        preds[row, :] = model.predict(val_X, verbose=1).ravel()
        actuals[row, :] = Y[ex, :, :].ravel()

    for r in range(preds.shape[0]):
        assert np.any(np.isfinite(preds[r,:]))
        assert np.any(np.isfinite(actuals[r, :]))

    return preds, actuals, selected_mins


def yhat_y_gen(summary, val_example_i, X, Y, model, len_in):
    val_examples = summary[val_example_i:, 0]
    val_minutes = summary[val_example_i:, 1]

    num_exs = val_examples.shape[0]
    if num_exs == 0:
        #raise ValueError('hey num_ex is 0')
        val_examples = summary[val_example_i-1:, 0]
        val_minutes = summary[val_example_i-1:, 1]
        num_exs = val_examples.shape[0]
        if num_exs == 0:
            raise ValueError('hey num_ex is 0')
    while True:
        rand_int = random.randint(0, num_exs - 1)
        rand_idx, rand_minutes = val_examples[rand_int], val_minutes[rand_int]
        val_X = np.array(X[rand_idx, :, :]).reshape((1, len_in, 1))
        predicted_Y = model.predict(val_X, verbose=1).ravel()
        val_Y = Y[rand_idx, :, :].ravel()
        yield predicted_Y, val_Y, rand_minutes


def single_device_examples_name(sid, label, dd):
    file_name = '_'.join([label, '{:03d}'.format(sid) + '.npy'])
    return os.path.join(dd, file_name)


def store_model_results_without_weights(results):
    results_with_lists = {}
    for k, v in results.items():
        val_items = []
        for klabel, vals in v.items():
            if isinstance(vals, np.ndarray):
                val_items.append((klabel, list(vals)))
            else:
                val_items.append((klabel, vals))
        results_with_lists[k] = dict(val_items)
    return results_with_lists


def init_example_args(X=None, len_in=None, len_out=None,
                      num_features=None):
    len_of_X = len(X[0,:])
    len_in_seq = len_in
    len_out_seq = len_out
    num_features = X.shape[0]
    num_examples = int(len_of_X/len_in_seq)
    return num_features


def rand_slices_every_five(rand_ys, len_in, md):
    idx = pd.IndexSlice
    rands = np.stack(tuple(rand_ys))
    for i in range(0, len_in, 5):
        if len(rands.shape) == 4:
            rand_slice = rands.loc[idx[:, :, i, :]]
            file_prefix = 'rand_yhats'
        elif len(rands.shape) == 3:
            rand_slice = rands.loc[idx[:, i, :]]
            file_prefix = 'rand_ys'
        f = os.path.join(md, file_prefix + '_' + str(i) + '.csv')
        np.savetxt(f, rand_slice.flatten())


def attention_model(X, Y, len_in, len_out, val_split=0.30, epochs=None,
                    patience=PATIENCE, dev_id=None, batch_size=None,
                    weights_suffix=None, weights_dir=None, optimizer=None,
                    checkpoint_callbacks=False):
    print(f'Version of tf: {tf.__version__}')
    print(f'Version of keras: {keras.__version__}')

    logger.info('begin attention fit for id %s', dev_id)
    len_per_example = len_in + len_out
    assert len_in == len_out
    model = Sequential()
    model.add(LSTM(len_in,
                   input_shape=(len_in, 1),
                   return_sequences=True))
    model.add(AttentionDecoder(len_in, 1))
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = AccHistory()

    num_epochs_patience = '_'.join([str(epochs), 'epochs',
                                    str(patience), 'patience'])
    epoch_val_loss = '.{epoch:02d}-{val_loss:.2f}'
    ext = '.h5'
    in_out = '_'.join(['in', str(len_in), 'out', str(len_out)])

    callbacks = [earlystop, history]
    if checkpoint_callbacks:
        checkpoint = '_'.join(['checkpoint', in_out, weights_suffix,
                               num_epochs_patience, epoch_val_loss,
                               str(dev_id) + ext])
        checkpointpath = os.path.join(weights_dir, checkpoint)
        checkpoints = ModelCheckpoint(checkpointpath, monitor='val_loss',
                                      verbose=0, save_best_only=True,
                                      save_weights_only=True)
        callbacks.append(checkpoints)

    X_encoded = X
    print('shape of X_encoded: {}'.format(X_encoded.shape))

    Y_encoded = Y

    print('shape of Y_encoded: {}'.format(Y_encoded.shape))

    # Added steps_per_epoch
    print(f'type of X_encoded: {type(X_encoded)}')
    print(f'type of X_encoded.shape: {type(X_encoded.shape)}')
    print(f'value of X_encoded.shape: {X_encoded.shape}')
    steps_per_epoch = max(1, int(int(X_encoded.shape[0]) / batch_size))
    print(f'steps_per_epoch: {steps_per_epoch}')
    hist = model.fit(X_encoded, Y_encoded, callbacks=callbacks,
                     epochs=epochs, steps_per_epoch=steps_per_epoch,
                     verbose=2)

    scores = model.evaluate(X_encoded, Y_encoded, verbose=0)

    metrics = model.metrics_names

    print('metrics names:', metrics)

    results = '%s: %.2f%%' % (metrics[1], scores[1]*100)

    print('results:', results)

    val_loss_list = hist.history['val_loss']
    loss_iter = (round(n, 5) for n in val_loss_list)
    loss_arr = np.fromiter(loss_iter, np.float32, count=len(val_loss_list))

    all_history = (model, model.metrics_names, scores,
                   'history_acc', history.accuracy,
                   'history_val_acc', history.val_accuracy,
                   'history_epoch_acc', history.epoch_accuracy,
                   'history_loss', hist.history['loss'],
                   'history_val_loss', loss_arr,
                   'patience', patience,
                   'stopped_epoch', earlystop.stopped_epoch,
                   'max_epochs', epochs)
    logger.info('end attention fit for id %s', dev_id)
    return all_history


def LSTM_model(X, Y, len_in, len_out, val_split=0.30, epochs=None,
               patience=PATIENCE, dev_id=None, batch_size=None,
               weights_suffix=None, weights_dir=None, optimizer=None,
               checkpoint_callbacks=False):
    logger.info('begin lstm fit for id %s', dev_id)
    len_per_example = len_in + len_out
    model = Sequential()
    model.add(LSTM(len_per_example,
                   input_shape=(len_in, 1),
                   return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])

    old_history = AccHistory()

    num_epochs_patience = '_'.join([str(epochs), 'epochs',
                                    str(patience), 'patience'])
    epoch_val_loss = '.{epoch:02d}-{val_loss:.2f}'
    ext = '.h5'
    in_out = '_'.join(['in', str(len_in), 'out', str(len_out)])

    # callbacks = [earlystop, old_history]
    callbacks = [earlystop]
    if checkpoint_callbacks:
        checkpoint = '_'.join(['checkpoint', in_out, weights_suffix,
                               num_epochs_patience, epoch_val_loss,
                               str(dev_id) + ext])
        checkpointpath = os.path.join(weights_dir, checkpoint)
        checkpoints = ModelCheckpoint(checkpointpath, monitor='val_loss',
                                      verbose=0, save_best_only=True,
                                      save_weights_only=True)
        callbacks.append(checkpoints)

    hist = model.fit(X, Y, callbacks=callbacks, validation_split=val_split,
                     batch_size=batch_size, epochs=epochs, verbose=2)

    scores = model.evaluate(X, Y, verbose=0)

    metrics = model.metrics_names

    print('metrics names:', metrics)

    results = '%s: %.2f%%' % (metrics[1], scores[1]*100)

    print('results:', results)

    val_loss_list = hist.history['val_loss']
    loss_iter = (round(n, 5) for n in val_loss_list)
    loss_arr = np.fromiter(loss_iter, np.float32, count=len(val_loss_list))

    # hist.history['accuracy'] replaces history.accuracy
    # hist.history['val_accuracy'] replaces history.val.accuracy
    # None replaces history.epoch_accuracy
    all_history = (model, model.metrics_names, scores,
                   'history_acc', hist.history['accuracy'],
                   'history_val_acc', hist.history['val_accuracy'],
                   'history_epoch_acc', None,
                   'history_loss', hist.history['loss'],
                   'history_val_loss', loss_arr,
                   'patience', patience,
                   'stopped_epoch', earlystop.stopped_epoch,
                   'max_epochs', epochs)
    logger.info('end lstm fit for id %s', dev_id)
    return all_history


def rand_slices_every_five(rand_ys, len_in, md):
    idx = pd.IndexSlice
    rands = np.stack(tuple(rand_ys))
    for i in range(0, len_in, 5):
        if len(rands.shape) == 4:
            rand_slice = rands.loc[idx[:, :, i, :]]
            file_prefix = 'rand_yhats'
        elif len(rands.shape) == 3:
            rand_slice = rands.loc[idx[:, i, :]]
            file_prefix = 'rand_ys'
        f = os.path.join(md, file_prefix + '_' + str(i) + '.csv')
        np.savetxt(f, rand_slice.flatten())


def summarize_aggregated_random_predictions(ids, data_dir=None, data='on_off',
                                            len_in=None, len_out=None,
                                            suffix=None, model_dir=None,
                                            references='ends', hour=None,
                                            spin_time=10, qmin=0.05,
                                            qmax=0.95, num_rands=2,
                                            write_examples=None,
                                            overwrite=None,
                                            weights_suffix=None):

    partial_rand_predict = partial(predict_random_examples, data_dir=data_dir,
                                   data=data, len_in=len_in, len_out=len_out,
                                   suffix=suffix,
                                   weights_suffix=weights_suffix,
                                   model_dir=model_dir,
                                   references=references, hour=hour,
                                   spin_time=spin_time, num_rands=num_rands,
                                   write_examples=write_examples,
                                   overwrite=overwrite)

    yhat_actuals = (partial_rand_predict(sid=sid) for sid in ids)
    yhat_actuals_with_exs = (yhats_sid for yhats_sid in yhat_actuals
                             if yhats_sid is not None)
    time_intervals = len_out + spin_time

    predict_arr = np.full((num_rands, time_intervals, len(ids)), np.nan)
    actual_arr = np.full(predict_arr.shape, np.nan)

    for dev, pred_actual in enumerate(yhat_actuals_with_exs):
        yhats, actuals = pred_actual
        predict_arr[:, :, dev] = yhats
        actual_arr[:, :, dev]= actuals

    poimeans = np.full((time_intervals, num_rands), np.nan)
    assert np.all(np.isnan(poimeans))
    actual_totals = np.full(poimeans.shape, np.nan)
    quantiles_min = np.full(poimeans.shape, np.nan)
    quantiles_max = np.full(poimeans.shape, np.nan)
    for n in range(num_rands):
        logging.debug('n in num_rands %s', n)
        for t in range(time_intervals):
            # devs = np.isfinite(predict_arr[:, t, n])
            devs = np.isfinite(predict_arr[n, t, :])
            if not any(devs):
                continue
            else:
                logging.debug('t in time_intervals %s', t)
                poimean, pp = poimean_int_probs(predict_arr, devs, t, n)
                # logging.debug('poimean from func %s', poimean)
                poimeans[t, n] = poimean
                # logging.info('poimean %s', poimean)
                # logging.info('pp %s', pp)
                quantile_min, quantile_max = quantiles_min_max(pp, qmin=qmin,
                                                               qmax=qmax)
                quantiles_min[t, n] = quantile_min
                quantiles_max[t, n] = quantile_max
                devs_on = np.sum(actual_arr[n, t, devs])
                actual_totals[t, n] = devs_on

    logging.info('now returning poimeans etc')
    logging.debug('about to return poimeans %s', poimeans)
    return poimeans, actual_totals, quantiles_min, quantiles_max


def poimean_int_probs(predict_arr, devs, t, n):
    #probs_t = predict_arr[devs, t, n]
    probs_t = predict_arr[n, t, devs]
    #range_arr = np.arange(probs_t.shape[0] + 1)
    range_arr = np.arange(probs_t.shape[0] + 1)
    kk = numpy2ri.py2ri(range_arr)
    pp = numpy2ri.py2ri(probs_t)
    pmf = poibin.dpoibin(kk=kk, pp=pp)
    poiweightedsum = np.multiply(np.array(pmf), range_arr)
    poimean = np.round(np.sum(poiweightedsum)).astype(np.int64)
    return poimean, pp


def quantiles_min_max(probs, qmin=0.05, qmax=0.95):
    qqnp = np.array([qmin, qmax])
    qqr = numpy2ri.py2ri(qqnp)
    quantile_min, quantile_max = poibin.qpoibin(qqr, probs)
    return quantile_min, quantile_max


def summarize_aggregated_errors(poimeans, actual_totals, spin_time, num_rands):
    divisor = np.sum(range(1, spin_time+1))
    weights_begin = 0.25 * array([minutes/divisor for minutes in
                                  range(1, spin_time+1)])
    weights_middle = 0.5 * np.ones(spin_time) / spin_time
    weights_end = 0.25 * array([numer/divisor for numer in
                                reversed(range(1, spin_time+1))])
    weights = np.concatenate((weights_begin, weights_middle, weights_end))

    weights.reshape((weights.size, 1))

    actuals = [actual_totals[:, n] for n in range(num_rands)]

    raw_residuals = [poimeans[:, n] - actual_totals[:, n]
                     for n in range(num_rands)]
    raw_abs_residuals = [np.abs(residuals) for residuals in raw_residuals]

    finite_ts = [np.isfinite(residuals) for residuals in raw_residuals]
    weighted_residuals = array([np.dot(weights[finite], res[finite])
                                for res, finite in zip(raw_residuals,
                                                       finite_ts)])
    weighted_abs_residuals = array([np.dot(weights[finite], res[finite]) for
                                    res, finite in zip(raw_abs_residuals,
                                                       finite_ts)])
    weighted_actuals = array([np.dot(weights[finite], actual[finite]) for
                              actual, finite in zip(actuals, finite_ts)])
    mean_abs_error = np.mean(weighted_abs_residuals)
    mean_error = np.mean(weighted_residuals)
    mean_actual = np.mean(weighted_actuals)

    relative_error = mean_error / mean_actual

    return mean_error, mean_abs_error, relative_error, mean_actual


def summarize_val_loss(val_losses):
    val_loss_avg = {}
    for k, v in val_losses.items():
        avg_val = np.mean(v[-20:])
        val_loss_avg[k] = avg_val
    val_loss_summ = [v for v in val_loss_avg.values()]
    val_loss_arr = np.array(val_loss_summ)
    valdf = pd.DataFrame(val_loss_arr)
    return valdf.quantile([0.25, 0.5, 0.75, .9])


def summarize_examples_and_training(ids, data_dir, suffix, model_dir,
                                    weights_suffix, len_in, len_out, val_split,
                                    ex_meta, train_meta, epochs, patience, ext,
                                    data='on_off'):
    ex_metas = {}
    for sid in ids:
        X, _, T = single_device_examples(sid, data_dir=data_dir, suffix=suffix,
                                         data=data, len_in=len_in,
                                         len_out=len_out)
        ex_metas[sid] = summarize_input(sid, X, T, len_in, val_split,
                                          data_dir, suffix)

    train_result = review_training_performance(ids, weights_suffix, model_dir,
                                               ext, len_in, len_out, epochs,
                                               patience)

    ex_train = []
    ex_train_idx = []
    for k, v in ex_metas.items():
        logging.debug('type of train_result[k] %s', type(train_result[k]))
        logging.debug('type of value %s', type(train_result[k][train_meta]))
        logging.debug('keys of value %s', type(train_result[k].keys()))
        ex_train.append((k, v[ex_meta], v['first'], v['last_train'],
                         len(v['ex_dates']),
                         np.mean(train_result[k][train_meta][-patience:])))
        ex_train_idx.append(k)

    col_names = ['dev', 'num_train', 'first', 'last', 'num_dates',
                 'avg_val_loss']

    ex_train_df = pd.DataFrame.from_records(ex_train, columns=col_names,
                                            index=ex_train_idx)
    ex_train_df.sort_values('num_dates', inplace=True)
    return ex_metas, train_result, ex_train_df


def summarize_previous_training(ids, model_dir, weights_suffix, ext, len_in,
                                len_out, epochs, patience):
    results = []
    for sid in ids:
        res_path = training_results_file(sid, weights_suffix, model_dir, ext,
                                         len_in, len_out, epochs, patience)
        losses = np.load(res_path)
        results.append((str(sid), losses['loss'], losses['val_loss']))

    return results


def plot_preds_actuals(len_out, means, quants_min, quants_max, actual_totals):
    ax = figure().gca()
    ax.set_xlabel('Minutes')
    ax.set_ylabel('ACs ON')

    ax.plot(np.arange(len(means)), means, label='Predicted')
    ax.fill_between(range(len(means)), quants_min, quants_max,
                     color='gray', alpha=0.4)
    ax.plot(np.arange(len(means)), actual_totals, label='Actual')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    show()


class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        #super(AttentionDecoder, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(st, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


def nn_main():
    parser = argparse.ArgumentParser(description='Cycling training')
    #parser.add_argument('--dev_mode', default=False, type=bool,
    #                    help='If True, only train for 2, not all')
    parser.add_argument('ids', type=str, default='100,102',
                        help='ids to train')
    #parser.add_argument('--datapkg', default='nickpowersys/constant', type=str,
    #                    help='Quilt data package')
    parser.add_argument('--data', type=str, help='NPZ file')
    parser.add_argument('--model', default='attention', type=str)
    parser.add_argument('--weightssuffix', type=str,
                        default='end_ex_off_profile',
                        help='description of training variant')
    parser.add_argument('--val_split', default=0.3, type=float,
                        help='percent for validation')
    parser.add_argument('--batchsize', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='epochs without improvement before stopping')
    parser.add_argument('--checkpoints', type=bool, default=False,
                        help='create checkpoints (Boolean)')
    parser.add_argument('--weightsdir', type=str, default='/artifacts',
                        help='directory for saving weights')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda?')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer model option')
    parser.add_argument('--tfseed', type=int, default=2,
                        help='random seed for TF to use. Default=2')
    parser.add_argument('--npseed', type=int, default=1,
                        help='random seed for np to use. Default=1')
    opt = parser.parse_args()

    dd = '/Users/everyday/Documents/Pachyderm/pachcaar/data/'
    md = '/Users/everyday/Documents/Pachyderm/pachcaar/models/'
    suffix = 'test_end_ex_off'
    weights_suffix = 'end_ex_off_profile'
    write_examples = False
    overwrite = False
    predict = True
    num_rands = 2
    len_in = 20
    len_out = 20
    spin_time = 10
    plot_sid = False
    val_split = 0.30

    # May want to log Exception if not automatically logged
    if opt.cuda and not tf.test.is_gpu_available(cuda_only=True):
        raise Exception("No GPU found, please run without --cuda")

    model_func = dict([('LSTM', LSTM_model),
                       ('attention', attention_model)])

    # single_device_examples(473, data_dir=dd)
    # ids = [473]
    # other_ids = [1, 219, 19]

    # [671] assert starts_temps_same.size > 0
    # [167,20,245,277,535,584] 0 samples
    # [443,445,524,557] attempt to get argmax of an empty sequence
    # [129, 143,108,23,291,33,35,425,432,441,472,485,506,517,58,6,92,93] assert all(times_until_ends[temps_decreasing] < 3600.0)
    # [119, 125,194,213,221,298,332,357,395,414,436,448,47,491,514,
    #  520,554,555] # arrays used as indices must be of integer (or boolean) type
    # ids = [100, 102, 106, 116,118,122,123,124,133,134,139,140,144,
    #        148, 154, 188, 2, 22, 223, 240, 275, 28, 284, 294, 313, 314, 324,
    #        330, 473, 339, 34, 345, 347, 358, 373, 392, 4, 409, 421, 422, 426,
    #        43, 434, 435, 444, 45, 46, 462, 473, 476, 480, 501, 504, 513, 516]
    # ids_lt_7_days = [118,585,513,409,521,140]

    # ids = [100,102,106,116,122,123,124,133,134,144,
    #        148,154,188,2,22,223,240,275,28,284,294,313,314,324,
    #        330,473,34,345,347,358,373,4,421,422,426,
    #        43,434,435,444,45,46,462,473,476,480,501,504,516,
    #        523,527,533,536,552,556,59,64,648,649,66,76,83,
    #        89,9,97,98,99]

    #ids_training_to_investigate = [339, 139, 392]
    # ids = [100,102,106,116,118]
    #ids = [100, 102]
    #ids = [100]

    #ids = [int(id) for id in opt.ids.split(',')]
    #ids = [556]
    print('ids: {}'.format(opt.ids))
    print('type of ids: {}'.format(type(opt.ids)))

    max_prev_jobs = 1

    completed = None

    #list_of tuples of devs, job ids
    #print('completed: {}'.format(completed))

    #print('about to start train')

    data_file = download_s3_file(opt.data, opt.data)
    print('type of data_file: ', type(data_file))
    data = get_xyt_arrs(data_file, ids=opt.ids)

    #assert 1 == 0
    print('opt.model: ', opt.model)
    print('model_func[opt.model]: ', model_func[opt.model])
    results = train(ids=opt.ids, data=data,
                    model=(opt.model, model_func[opt.model]),
                    val_split=opt.val_split, patience=opt.patience,
                    batch_size=opt.batchsize, epochs=opt.epochs,
                    weights_suffix=opt.weightssuffix,
                    weights_dir=opt.weightsdir, optimizer=opt.optimizer,
                    checkpoint_callbacks=opt.checkpoints,
                    completed=completed, ext='.npz')

    # print(results)
    #return results

    # results = train_and_summarize(ids, data_dir=dd, data='on_off',
    #                               suffix=suffix, weights_suffix=weights_suffix,
    #                               len_in=20, len_out=20, model_dir=md,
    #                               write_examples=write_examples,
    #                               overwrite=overwrite, predict=predict,
    #                               val_split=val_split, plot_sid=plot_sid,
    #                               epochs=opt.epochs, patience=PATIENCE,
    #                               weights_dir=md, summary_ext='.npz',
    #                               quilt_pkg=opt.datapkg)

    input_metas = results[1]

    return results


if __name__ == '__main__':
    ### BEGIN COPY FROM nn_main()
    parser = argparse.ArgumentParser(description='Cycling training')
    # parser.add_argument('--dev_mode', default=False, type=bool,
    #                    help='If True, only train for 2, not all')
    parser.add_argument('ids', type=str, default='100,102',
                        help='ids to train')
    # parser.add_argument('--datapkg', default='nickpowersys/constant', type=str,
    #                    help='Quilt data package')
    parser.add_argument('--data', type=str, help='NPZ file')
    parser.add_argument('--model', default='LSTM', type=str)
    parser.add_argument('--weightssuffix', type=str,
                        default='end_ex_off_profile',
                        help='description of training variant')
    parser.add_argument('--val_split', default=0.3, type=float,
                        help='percent for validation')
    parser.add_argument('--batchsize', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='epochs without improvement before stopping')
    parser.add_argument('--checkpoints', type=bool, default=False,
                        help='create checkpoints (Boolean)')
    parser.add_argument('--weightsdir', type=str, default='/artifacts',
                        help='directory for saving weights')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda?')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer model option')
    parser.add_argument('--tfseed', type=int, default=2,
                        help='random seed for TF to use. Default=2')
    parser.add_argument('--npseed', type=int, default=1,
                        help='random seed for np to use. Default=1')
    opt = parser.parse_args()

    # May want to log Exception if not automatically logged
    if opt.cuda and not tf.test.is_gpu_available(cuda_only=True):
        raise Exception("No GPU found, please run without --cuda")

    model_func = dict([('LSTM', LSTM_model),
                       ('attention', attention_model)])

    print('ids: {}'.format(opt.ids))
    print('type of ids: {}'.format(type(opt.ids)))

    completed = None

    data_file = download_s3_file(opt.data, opt.data)
    logger.debug('type of data_file: ', type(data_file))
    data = get_xyt_arrs(data_file, ids=opt.ids)

    print('opt.model: ', opt.model)
    print('model_func[opt.model]: ', model_func[opt.model])
    logger.info(f"model_func[opt.model]: , {model_func[opt.model]}")
    results = train(ids=opt.ids, data=data,
                    model=(opt.model, model_func[opt.model]),
                    val_split=opt.val_split, patience=opt.patience,
                    batch_size=opt.batchsize, epochs=opt.epochs,
                    weights_suffix=opt.weightssuffix,
                    weights_dir=opt.weightsdir, optimizer=opt.optimizer,
                    checkpoint_callbacks=opt.checkpoints,
                    completed=completed, ext='.npz')

    #return results
    options = results
    # ex_metas, train_result, ex_train_df, mean_error, mean_abs_error, relative_error, mean_actual, poimeans, quants_min, quants_max, actual_totals = main()
    # result, mean_error, mean_abs_error, relative_error, mean_actual = main()
    # results, profiled, profile_fit = main()
    #rand_predicts, actuals = main()

    #options = nn_main()

