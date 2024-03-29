import os
import json
import h5py

split_result_path = os.path.join('..', '..', 'dataset', 'split')
dirs_path_list = [os.path.join(split_result_path, x) for x in os.listdir(split_result_path) if os.path.isdir(os.path.join(split_result_path, x))]

for dir in dirs_path_list:
    files = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.json')]
    assert len(files) == 1
    with open(files[0]) as f:
        data = json.load(f)

hdf5_dir = os.path.join('..', '..', 'oxford_data')
hdf5_path = os.path.join(hdf5_dir, 'oxford.hdf5')

with h5py.File(hdf5_path, 'r') as f:
    for group in f['test']:
        print(f'group = {group}')