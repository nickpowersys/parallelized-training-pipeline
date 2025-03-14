import argparse
import inspect
import json
import logging.config
from pathlib import Path, PurePath
import re

import toml

import keras
from keras.layers import TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import model_from_json, Sequential

import numpy as np

import tensorflow as tf

from access import get_x_arrs
from accesscloud import download_s3_file

logging.config.fileConfig('./logging.ini', defaults={'logfilename': './mylog.log'},
                          disable_existing_loggers=True)
logger = logging.getLogger(__name__)

config = toml.load("./config.toml")

S3_FILE = config["data"]["data_file"]

NUM_FEATURES = config["models"]["num_features"]


def populate_arch_dict(memory_units, input_shape, activation):
    arch = lstm_architecture(memory_units, input_shape, activation=activation)
    arch_dict = dict()
    arch_dict['json_model'] = arch.to_json()
    arch_dict['model_source'] = layers_specification_source_code(lstm_architecture)
    arch_dict['keras_version'] = keras.__version__
    arch_dict['tf_version'] = tf.__version__
    arch_dict['memory_cells'] = memory_units
    arch_dict['input_shape'] = input_shape
    arch_dict['activation'] = activation
    return arch_dict


def lstm_architecture(memory_cells, input_shape, activation=None):
    if activation is None:
        activation = config["tf"]["activation"]

    model = Sequential()
    # Number of memory cells defines length of this fixed-size [output] vector.
    # Brownlee, 9.2.4
    model.add(LSTM(memory_cells, input_shape=input_shape, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation=activation)))
    return model


def load_architecture_from_json_file(json_file, model_dir=None):
    if model_dir is None:
        model_dir = config["models"]["model_dir"]
    model_path = PurePath(model_dir).joinpath(json_file)
    if Path(model_path).exists():
        with open(model_path, 'r') as opened:
            loaded_json = json.load(opened)
        loaded_dict = json.loads(loaded_json)
        loaded_dict['input_shape'] = tuple(loaded_dict['input_shape'])
        return ModelArchitecture(loaded_dict)
    else:
        raise ValueError('There is no file {}'.format(model_path))


def memory_cells_as_sum_len_in_len_out(len_in, len_out):
    return len_in + len_out


class ModelArchitecture:
    def __init__(self, model_and_metadata):
        self.model = model_from_json(model_and_metadata['json_model'])
        self.model_source = model_and_metadata['model_source']
        self.keras_version = model_and_metadata['keras_version']
        self.tf_version = model_and_metadata['tf_version']

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_object):
        self._model = model_object
        self._memory_cells = self._get_config_attr(layer=0, attr='units')
        self._input_shape = self._get_config_attr(layer=0, attr='batch_input_shape')
        self._activation = self._get_config_attr(layer=1, attr='activation')

    @property
    def model_source(self):
        return self._model_source

    @model_source.setter
    def model_source(self, model_code):
        self._model_source = model_code

    @property
    def keras_version(self):
        return self._keras_version

    @keras_version.setter
    def keras_version(self, version):
        self._keras_version = version

    @property
    def tf_version(self):
        return self._tf_version

    @tf_version.setter
    def tf_version(self, version):
        self._tf_version = version

    @property
    def memory_cells(self):
        return self._memory_cells

    @memory_cells.setter
    def memory_cells(self, units):
        if not isinstance(units, int):
            raise ValueError('Memory_cells must be an int.')
        self._memory_cells = units

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape):
        if (not isinstance(shape, tuple) or not isinstance(shape[1], int) or
                not isinstance(shape[2], int)):
            raise ValueError('Input_shape from config must be 3-tuple of ints')
        self._input_shape = shape[1], shape[2]

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, activation):
        self._activation = activation

    def _get_config_attr(self, layer: int, attr: str):
        layer_config = self.model.layers[layer].get_config()
        # Check if layer nested in layer
        if 'layer' in layer_config:
            nested_layer_config = layer_config['layer']['config']
            config_attr = nested_layer_config[attr]
        else:
            config_attr = layer_config[attr]
        return config_attr

    def __repr__(self):
        model_and_source = {'json_model': self.model.to_json(),
                            'model_source': self.model_source,
                            'keras_version': self.keras_version,
                            'tf_version': self.tf_version,
                            'memory_cells': self.memory_cells,
                            'input_shape': self.input_shape,
                            'activation': self.activation}
        return f'{__class__.__name__}(' + repr(model_and_source) + ')'

    def __eq__(self, other):
        model = self.model
        other_model = other.model
        all_layers_match = True
        non_matching_layers = []
        for i, a, b in zip(range(len(model.layers)), model.layers, other_model.layers):
            self_config, other_config = a.get_config(), b.get_config()
            for k, v in self_config.items():
                if k != 'name' and v != other_config[k]:
                    all_layers_match = False
                    log_msg = 'Layer {}, Attribute {}: {} != {} (container != local).'
                    non_matching_layers.append(log_msg.format(i, k, v, other_config[k]))

        non_matching_attrs = []
        for attr in (k for k in self.__dict__.keys() if k != '_model'):
            if self.__dict__[attr] != other.__dict__[attr]:
                non_matching_attrs.append((attr, self.__dict__[attr], other.__dict__[attr]))

        if all_layers_match:
            logger.info('All layers match.')
        else:
            for n in non_matching_layers:
                logger.error(n)

        if non_matching_attrs:
            for attr in non_matching_attrs:
                logger.error(f'{attr[0]}: {attr[1]} != {attr[2]}')
            return False

        if all_layers_match:
            return True
        else:
            return False

    def write_architecture_to_json_file(self, json_file, model_dir=None):
        if model_dir is None:
            model_dir = config["models"]["model_dir"]
        model_and_source = {'json_model': self.model.to_json(),
                            'model_source': self.model_source,
                            'keras_version': self.keras_version,
                            'tf_version': self.tf_version,
                            'memory_cells': self.memory_cells,
                            'input_shape': self.input_shape,
                            'activation': self.activation}
        model_path = PurePath(model_dir).joinpath(json_file)
        if not Path(model_path).exists():
            json_str = json.dumps(model_and_source, indent=4)
            with open(model_path, "w") as f:
                json.dump(json_str, f)
                print('{} written. Keras {} Tensorflow {}'.format(model_path,
                                                                  self.keras_version,
                                                                  self.tf_version))
        else:
            raise ValueError('There is a file with the name {}'.format(json_file))


def layers_specification_source_code(model_func):
    model_spec = inspect.getsourcelines(model_func)[0]
    first_code_line = [i for i, line in enumerate(model_spec)
                       if 'Sequential()' in line and not re.search(r'^\s+\#', line)][0]
    code_lines = [line for line in model_spec[first_code_line:] if
                  not re.search(r'^\s+\#', line)]
    model_source = ''.join(code_lines)
    return model_source


def compile_model(model, optimizer=None):
    if optimizer is None:
        optimizer = config["tf"]["optimizer"]
    loss = config["tf"]["loss"]
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def main():
    # File name, such as: 'lstm.json'
    json_file = config["tf"]["model_json"]
    # quilt_pkg = config["quilt"]["user"] + '/' + config["quilt"]["package"]
    model_dir = config["models"]["model_dir"]

    activation = config["tf"]["activation"]

    memory_cells_to_exec = config["models"]["memory_units"]

    #data, dev_keys, metadata = quiltdata.get_data_set(quilt_pkg)
    #len_in, len_out = quiltdata.len_of_x_and_y(data, dev_keys)
    data_file = download_s3_file(S3_FILE)
    loaded = np.load(data_file)
    len_in, len_out, dev_keys = get_x_arrs(loaded)

    memory_cells = memory_cells_as_sum_len_in_len_out(len_in, len_out)

    parser = argparse.ArgumentParser(description='Model specification')
    parser.add_argument('json_file', type=str, nargs='?', default=json_file,
                        help='JSON file for model as json and metadata')
    # parser.add_argument('data_package', type=str, nargs='?', default=quilt_pkg,
    #                    help='Quilt data package')
    parser.add_argument('model_dir', type=str, nargs='?', default=model_dir,
                        help='Directory for writing model')
    parser.add_argument('memory_cells', type=int, nargs='?', default=memory_cells,
                        help='Number of memory cells')
    parser.add_argument('input_shape', type=tuple, nargs='?',
                        default=(len_in, NUM_FEATURES),
                        help='Input shape (Vector size, features)')
    parser.add_argument('activation', type=str, nargs='?', default=activation,
                        help='Activation')
    opts = parser.parse_args()

    arch_dict = populate_arch_dict(opts.memory_cells, opts.input_shape,
                                   opts.activation)
    ma = ModelArchitecture(arch_dict)

    ma.compile_model(model=ma.model)

    return ma


if __name__ == '__main__':
    architecture = main()
