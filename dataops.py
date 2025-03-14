import glob
import os

import numpy as np

os.chdir("/Users/everyday/Documents/Pachyderm/pachcaar/data/train")

all_files = glob.glob("*.npy")

npy_arrs = []
for f in all_files:
    npy_arrs.append(np.load(f))

arr_dict = {}
for fn, arr in zip(all_files, npy_arrs):
    arr_dict[fn] = arr

np.savez("all_xyt_74ids.npz", **arr_dict)
