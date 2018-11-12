import numpy as np

def fix_nans(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def pad_array(array, max_len):
    return np.asarray([np.pad(a, [(max_len - len(a), 0), (0,0)], mode='constant') for a in array], dtype=np.float32)

def limit_length_and_pad(prim, evo, dih, max_length):
    mask = np.array([len(el) for el in prim]) <= max_length
    prim_lim, evo_lim, dih_lim = np.array(prim)[mask], np.array(evo)[mask], np.array(dih)[mask]
    prim_pad, evo_pad, dih_pad = pad_array(prim_lim, max_length), pad_array(evo_lim, max_length), pad_array(dih_lim, max_length)
    return fix_nans(prim_pad), fix_nans(evo_pad), fix_nans(dih_pad)