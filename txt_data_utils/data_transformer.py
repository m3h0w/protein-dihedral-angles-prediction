import numpy as np

def fix_nans(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def pad_array(array, max_len=None, value=0., dtype=np.float32):
    print("padded")
    if(len(array[0].shape) < 2):
        array = np.array([a.reshape(-1,1) for a in array])
    print(array.shape, array[0].shape)
    if max_len == None:
        max_len = np.max([len(a) for a in array])
    return np.asarray([np.pad(a, [(max_len - len(a), 0), (0,0)], mode='constant', constant_values=value) for a in array], dtype=dtype)

def limit_length_and_pad(prim, evo, dih, mask, max_length=None):
    pri_lengths = [len(el) for el in prim]
    if max_length == None:
        max_length = np.max(pri_lengths)

    len_mask = np.array(pri_lengths) <= max_length
    prim_lim, evo_lim, dih_lim, mask_lim = (np.array(prim)[len_mask], np.array(evo)[len_mask], 
                                            np.array(dih)[len_mask], np.array(mask)[len_mask])
    prim_pad, evo_pad, dih_pad, mask_pad = (pad_array(prim_lim, max_length), pad_array(evo_lim, max_length), 
                                            pad_array(dih_lim, max_length), pad_array(mask_lim, max_length, value=False, dtype=np.bool))
    mask_pad = mask_pad.reshape(mask_pad.shape[0], -1) # this is necessary because numpy expects this shape to use mask as an index
    return fix_nans(prim_pad), fix_nans(evo_pad), fix_nans(dih_pad), fix_nans(mask_pad)