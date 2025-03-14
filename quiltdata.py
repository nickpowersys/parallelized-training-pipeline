from numpy import array
import quilt


def install_data(pkg):
    # force to avoid y/n prompt; does not re-download
    quilt.install(pkg, force=True)


def get_data_set(pkg, debug=True):
    if not debug:
        install_data(pkg)
    loaded_pkg = quilt.load(pkg)
    g_keys = loaded_pkg._group_keys()
    dev_keys = [k for k in g_keys if k != 'metadata']
    return loaded_pkg, dev_keys, loaded_pkg['metadata']


def len_of_x_and_y(data, dev_keys):
    sample_sid = dev_keys[0]
    sample_data = data[sample_sid]
    random_X, random_Y = sample_data.X(), sample_data.Y()
    len_in, len_out = random_X.shape[1], random_Y.shape[1]
    return len_in, len_out


def ids_remaining_to_train(train_ids, dev_keys, completed):
    """Determine """
    previous_jobs_exist = True if completed else False
    if previous_jobs_exist:
        print('completed: {}'.format(completed))
        devs_jobs = dict(completed)
        devs_trained = devs_jobs.keys()
        to_complete = set(dev_keys) - set(devs_trained)
    else:
        to_complete = set(dev_keys)
    if train_ids is not None:
        dev_ids = set(train_ids.split(','))
        to_complete = to_complete.intersection(dev_ids)
        print('to complete: {}'.format(to_complete))

    to_complete = list(to_complete)
    mod_ids = add_prefix_for_quilt(to_complete)
    return to_complete, mod_ids

def add_prefix_for_quilt(ids):
    return ['n' + id for id in ids]


def remove_prefix_for_quilt_data(ids):
    return [id[1:] for id in ids]


def load_arrays_for_id(dev_id, data):
    sidnode = data[dev_id]
    X, Y, T = sidnode.X(), sidnode.Y(), sidnode.T()
    return X, Y, T


def summarize_input_for_all(user, package, data, dev_keys, val_split):
    len_in, len_out = quiltdata.len_of_x_and_y(data, dev_keys)
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
