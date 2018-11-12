from tqdm import tqdm
from sklearn.externals import joblib
import numpy as np

DATA_ORDER = ["[ID]", "[PRIMARY]", "[EVOLUTIONARY]", "[TERTIARY]", "[MASK]"]

def count_protein(raw_txt_data):
    data = filter_line_end(raw_txt_data)
    result = 0
    for line in data:
        if line == DATA_ORDER[0]: #id
            result += 1
    return result

def filter_line_end(data):
    return [str_.replace('\n', '') for str_ in data]

def get_primary_from_all_data(data, lim):
    result = []
    protein_count = 0
    flag = False
    for line in data:
        if line == DATA_ORDER[2]: #evolutionary
            flag = False
        if flag:
            result.append(line)
        if line == DATA_ORDER[1]: #primary
            flag = True
        if line == DATA_ORDER[0]: #id
            protein_count += 1
        if lim and protein_count > lim:
            return result
    return result

def get_evolutionary_from_all_data(data, lim):
    result = []
    protein_count = 0
    flag = False
    for line in data:
        if line == DATA_ORDER[-2]: #mask
            flag = False
        if flag:
            result.append(line)
        if line == DATA_ORDER[-3]: #tertiary
            flag = True
        if line == DATA_ORDER[0]: #id
            protein_count += 1
        if lim and protein_count > lim:
            return result
    return result

def get_tertiary_from_all_data(data, lim):
    result = []
    protein_count = 0
    flag = False
    for line in data:
        if line == DATA_ORDER[-1]: #mask
            flag = False
        if flag:
            result.append(line)
        if line == DATA_ORDER[-2]: #tertiary
            flag = True
        if line == DATA_ORDER[0]: #id
            protein_count += 1
        if lim and protein_count > lim:
            return result
    return result

def group_aminoacids_together(data, every_n):
    data_expanded = [np.asarray(np.expand_dims(t.split('\t'), 1), dtype=np.float32) for t in data]
    result = []
    for i in tqdm(range(0,len(data_expanded),every_n)):
        # group together every_n entries (e.g. 3 for tertiary and 21 for evo)
        result.append(np.concatenate([data_expanded[i+r] for r in range(every_n)], axis=1))
        
    return result

def parse_tertiary_from_file(path, data_lim=None):
    with open(path) as f:
        data = f.readlines()
    
    data_ = filter_line_end(data[:data_lim])        
    only_tertiary = get_tertiary_from_all_data(data_, data_lim)
    return group_aminoacids_together(only_tertiary, every_n=3)

def parse_evolutionary_from_file(path, data_lim=None):
    with open(path) as f:
        data = f.readlines()

    data_ = filter_line_end(data)
    print("Loaded data and filtered line endings")
    only_evo = get_evolutionary_from_all_data(data_, data_lim)
    print("Extracted evolutionary data")
    res = group_aminoacids_together(only_evo, every_n = 21)
    print("Grouped 21's together")
    return res

def parse_primary_from_file(path, data_lim=None):
    with open(path) as f:
        data = f.readlines()
    
    data_ = filter_line_end(data)
    print("Loaded data and filtered line endings")
    primary = get_primary_from_all_data(data_, data_lim)
    print("Extracted primary data")

    try:
        le = load_file('le.joblib')
    except Exception as e:
        print(e, "You need to have the Label Encoder 'le.joblib' in the same folder as your scripts")

    try:
        ohe = load_file('ohe.joblib')
    except Exception as e:
        print(e, "You need to have the One Hot Encoder 'ohe.joblib' in the same folder as your scripts")

    primary_in_floats = [le.transform([_ for _ in c]) for c in primary]
    primary_encoded = [ohe.transform(a.reshape(-1,1)).toarray() for a in primary_in_floats]
    print("Encoded primary sequences")
    return primary_encoded

def get_dih(protein_tertiary):
    p = protein_tertiary
    r = p.shape[0]
    a_list = list(range(r))
    the_list = np.array([a_list[slice(i, i+4)] for i in range(r - 4+1)])
    slices = np.asarray(p[the_list], dtype=np.float32)
    one_dih = np.array([dihedral(slice_) for slice_ in slices])
    one_dih = np.insert(one_dih, 0, None)
    one_dih = np.append(one_dih, [None,None])
    return one_dih.reshape(-1,3)

def save_file(data, path):
    joblib.dump(data, path) 
    
def load_file(path):
    return joblib.load(path)