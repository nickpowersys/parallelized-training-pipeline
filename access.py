import logging
import os
from pathlib import Path, PurePath
import pandas as pd
import re

import dask.bag as db
import numpy as np
from numpy import array
import toml

from accesscloud import download_s3_file

#logging.config.fileConfig('./logging.ini', disable_existing_loggers=False)
#logger = logging.getLogger(__name__)

# Load non-sensitive environment variables
config = toml.load("./config.toml")

S3_FILE = config["data"]["data_file"]


def arrs_by_id(data_dir, globstr, get_id_func=None, get_xyt_func=None):
    """Get NumPy paths"""
    paths = Path(data_dir).glob(globstr)

    b = db.from_sequence(paths)
    ids_xyt_paths = b.map(
        lambda p: (get_id_func(Path(p).name), get_xyt_func(Path(p).name), p)
    )

    x_ids = _ids_for_xyt_kind(ids_xyt_paths, xyt_kind="x")
    y_ids = _ids_for_xyt_kind(ids_xyt_paths, xyt_kind="y")
    t_ids = _ids_for_xyt_kind(ids_xyt_paths, xyt_kind="t")

    ids = set.intersection(x_ids, y_ids, t_ids)
    assert len(ids) == max(len(x_ids), len(y_ids), len(t_ids))
    paths_incl_xyt = ids_xyt_paths.filter(lambda x: x[0] in ids)
    x_files, y_files, t_files = dict(), dict(), dict()
    xyt_id_paths = {"x": x_files, "y": y_files, "t": t_files}
    for dev_id, x_y_or_t_kind, abs_path in paths_incl_xyt:
        xyt_id_paths[x_y_or_t_kind][dev_id] = str(abs_path)
    # return ids, xyt_id_paths
    return xyt_id_paths


#def xyt_arrs_to_save_to_npz(ids: set, arrs: dict):
def xyt_arrs_to_save_to_npz(arrs: dict):
    # Create a dict for saving the npz
    xyt_ids = dict()
    # Keys are just x, y, z
    for xyt, xyt_id_paths in arrs.items():
        # Keys are numerical
        for train_id, full_path in xyt_id_paths.items():
            # Combine the keys
            xyt_ids["_".join([xyt, train_id])] = np.load(full_path)
    # assert len(xyt_ids) == len(ids) * 3
    return xyt_ids


def get_id(path: str):
    return re.search(r"\d+", path).group()


def get_xyt(path: str):
    return path[0].lower()


def data_path():
    if 'test_data_file' in os.environ:
        path = os.getenv('test_data_file')
    else:
        path = download_s3_file(S3_FILE)
    return path


def get_x_arrs(loaded_npz):
    xyt_ids = set(loaded_npz.keys())
    x_arrs = [x_id for x_id in xyt_ids if x_id[0] == "x"]
    sample_x_arr = x_arrs[0]
    sample_y_arr = str(x_arrs[0]).replace("x", "y")
    len_in = loaded_npz[sample_x_arr].shape[1]
    len_out = loaded_npz[sample_y_arr].shape[1]
    dev_keys = [x_id.split("_")[1] for x_id in x_arrs]
    return len_in, len_out, dev_keys


def get_s3_file(S3_FILE):
    data_file = download_s3_file(S3_FILE)
    return data_file


def get_xyt_arrs(npz_file, ids=None, sep=','):

    loaded = np.load(npz_file)
    if ids is not None:
        xyt_ids = set([xyt for xyt in loaded.files if xyt.split('_')[1] in ids])
    else:
        xyt_ids = loaded.files
    x_ids = set([i.split('_')[1] for i in xyt_ids if i[0] == 'x'])
    y_ids = set([i.split('_')[1] for i in xyt_ids if i[0] == 'y'])
    t_ids = set([i.split('_')[1] for i in xyt_ids if i[0] == 't'])
    xyt_ids = t_ids.intersection(x_ids, y_ids)
    ids_missing_from_data = set([i for i in ids.split(sep)
                                 if i not in xyt_ids])
    if ids_missing_from_data:
        msg = 'These IDs are missing from data: {}'
        raise ValueError(msg.format(set(ids_missing_from_data)))
    data_by_id = dict([(id, dict([(xyt, loaded[xyt + '_' + id])
                                  for xyt in ['x', 'y', 't']]))
                        for id in xyt_ids])
    return data_by_id


def summarize_input(
    sid, X, T, len_in, val_split, data_dir=None, suffix=None, write=False
):
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

    ex_date_arr = pd.DatetimeIndex(list(ex_dates)).values.astype("datetime64[D]")

    if write:
        input_file = "_".join(["input", suffix, str(sid)])
        input_path = os.path.join(data_dir, input_file)
        try:
            np.savez_compressed(
                input_path,
                first=array([first]),
                last=array([last]),
                last_train=array([last_train]),
                ex_per_day=array([ex_per_day]),
                num_train=array([num_train]),
                ex_dates=ex_date_arr,
            )
            logger.info("%s written.", input_path)
        except IOError:
            logger.info("Unable to write to %s", input_path)

    input_meta = dict(
        [
            ("first", first),
            ("last", last),
            ("last_train", last_train),
            ("ex_per_day", ex_per_day),
            ("num_train", num_train),
            ("ex_dates", ex_dates),
        ]
    )
    return input_meta


def npz_equal(npz_one, npz_two):
    loaded_one, loaded_two = np.load(npz_one), np.load(npz_two)
    keys_one, keys_two = set(loaded_one.keys()), set(loaded_two.keys())

    if keys_one != keys_two:
        return False
    for k in keys_one:
        if not np.array_equal(loaded_two[k], loaded_one[k]):
            return False
    return True


def _ids_for_xyt_kind(xyt_paths: db.Bag, xyt_kind: str):
    paths = xyt_paths.filter(lambda x: x[1] == xyt_kind)
    ids = set(paths.map(lambda x: x[0]))
    assert len(list(paths)) == len(ids)
    return ids
