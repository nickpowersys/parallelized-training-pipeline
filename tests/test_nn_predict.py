from unittest.mock import Mock
import pytest

import keras.models
import numpy as np

from nn_predict import LSTM_model


@pytest.fixture
def x_np_3d():
    return np.ones((3,2,1), dtype=float)


@pytest.fixture
def y_np_3d():
    return np.ones((3,2,1), dtype=float)


@pytest.fixture
def len_in():
    return 20


@pytest.fixture
def len_out():
    return 20


@pytest.fixture
def val_split():
    return 0.30


@pytest.fixture
def epochs():
    return 3


@pytest.fixture
def patience():
    return 100


@pytest.fixture
def dev_id():
    return '333'


@pytest.fixture
def batch_size():
    return 100


@pytest.fixture
def weights_suffix():
    return '_weight_suffix'


@pytest.fixture
def weights_dir():
    return '/weights/dir'


@pytest.fixture
def optimizer():
    return 'adam'


@pytest.fixture
def callbacks():
    return True


@pytest.fixture
def lstm_model_args(x_np_3d, y_np_3d, len_in, len_out,
                    val_split, epochs, patience, dev_id, batch_size,
                    weights_suffix, weights_dir, optimizer, callbacks):
    lstm_args_labeled = {'X': x_np_3d, 'Y': y_np_3d, 'len_in': len_in,
                         'len_out': len_out}
    lstm_args = [v for v in lstm_args_labeled.values()]
    lstm_kwargs = {'val_split': val_split, 'epochs': epochs,
                   'patience': patience, 'dev_id': dev_id, 'batch_size': batch_size,
                   'weights_suffix': weights_suffix, 'weights_dir': weights_dir,
                   'optimizer': optimizer, 'checkpoint_callbacks': callbacks}
    yield lstm_args, lstm_kwargs


def test_lstm_model_sequential_model(monkeypatch, lstm_model_args):
    def mock_sequential():
        return keras.models.Sequential()

    # custom class to be the mock return value
    # will override the requests.Response returned from requests.get
class MockSequential:
    # mock json() method always returns a specific testing dictionary
    #def __init__(self):
    #    # bare init method
    #    m = Mock()
    #    return m

    # requests.get is like keras.models Sequential
    # requests.get can further call .json() method
    # Sequential can further call .add method
    # function get_json calls the real json method
    # the static method of mockresponse, json, monkeypatches json
    # by returning a variable

    # In place of r, which calls json, I need
    # model, which is a Mock
    # i need that model to be returned when sequentuial is called

    # r = requests.get(url) -> model = keras.models.Sequential()
    # r.json() -> model.add

    model = Mock() # can mock response return this? without involving a mocksequential class?
    # could do
    # model could be a magicmock too
    mock.add = Mock()

    mock.add.side_effect = add_side_effect

    @staticmethod
    def add():
        a = Mock()
        a.side_effect = [5, 10]




    @staticmethod
    def add():
        # Should it return or hold a side eeffect?
        # return {"mock_key": "mock_response"}

    def test_sequential_called(monkeypatch):
        # Any arguments may be passed and mock_get() will always return our
        # mocked object, which only has the .json() method.
        def mock_sequential(*args, **kwargs):
            return MockSequential()

        # apply the monkeypatch for requests.get to mock_get
        #monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(keras.models, "Sequential", mock_sequential)

        # LSTM_model is user defined func

        # Sequential is method of object that the user-defined object calls

        monkeypatch.setattr(keras.models, "Sequential", mock_sequential)
        args, kwargs = lstm_model_args
        ex_lstm_out = LSTM_model(*args, **kwargs)
        keras.models.Sequential.assert_called_once()

