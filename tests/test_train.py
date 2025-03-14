from pathlib import PurePath

import pytest

from train import attention_model, LSTM_model, train

from settings import LOCAL_DATA_DIR


@pytest.fixture
def test_npz_file():
    return str(PurePath(LOCAL_DATA_DIR).joinpath('beca58all_xyt_74ids.npz'))


@pytest.mark.parametrize("ids, model, model_dir, model_json, val_split,"
                         "patience, batch_size, epochs, weights_suffix, weights_dir,"
                         "optimizer, checkpoints, completed, test_data_file",
                         [("102,648", "LSTM", "./models", "lstm.json", 0.3, 20, 20,
                          1000, "test_weights_suffix", "/artifacts", "adam", False,
                          None, pytest.lazy_fixture('test_npz_file'))])
def test_select_clean_auto(monkeypatch, ids, model, model_dir, model_json, val_split, patience,
                           batch_size, epochs, weights_suffix, weights_dir, optimizer,
                           checkpoints, completed, test_data_file):
    monkeypatch.setenv('test_data_file', test_data_file)
    model_func = dict([('LSTM', LSTM_model),
                       ('attention', attention_model)])

    results = train(train_ids=ids,
                    model=(model, model_func[model]),
                    model_dir=model_dir, model_json=model_json,
                    val_split=val_split, patience=patience,
                    batch_size=batch_size, epochs=epochs,
                    weights_suffix=weights_suffix,
                    weights_dir=weights_dir, optimizer=optimizer,
                    checkpoint_callbacks=checkpoints,
                    completed=completed, ext='.npz')
    monkeypatch.delenv('test_data_file')
    assert isinstance(results, tuple)
