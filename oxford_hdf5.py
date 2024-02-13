import os
import re
import cv2
import h5py
import json
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter

split_result_path = os.path.join('..', 'dataset', 'split')
dirs_path_list = [os.path.join(split_result_path, x) for x in os.listdir(split_result_path) if os.path.isdir(os.path.join(split_result_path, x))]

level_count = []
for dirs_path in dirs_path_list:
    level_count.append(int(dirs_path.split(os.sep)[-1][0]))
# 使用Counter统计每个元素出现的次数
element_count = Counter(level_count)
# 9个级别的绘本，每个级别训练集:验证集:测试集=5:1:1
for key, value in element_count.items():
    element_count[key] = [value-value//7*2, value//7, value//7]


train_dirs, valid_dirs, test_dirs = [], [], []
dirs_level = {key: [] for key in element_count.keys()}
for dirs_path in dirs_path_list:
    level = int(dirs_path.split(os.sep)[-1][0])
    dirs_level[level].append(dirs_path)

for key, value in dirs_level.items():
    train_dirs += value[ : element_count[key][0]]
    valid_dirs += value[element_count[key][0] : element_count[key][0]+element_count[key][1]]
    test_dirs += value[element_count[key][0]+element_count[key][1] : ]

def extract_number(s):
    return int(re.search(r'_([0-9]+).png', s).group(1))

def merge_dicts(dicts : list[dict])->dict:
    """
    将多个字典合并为一个字典，并检查键是否有冲突。
    
    :param dicts: 要合并的字典
    :return: 合并后的字典
    :raises ValueError: 如果在字典中发现键冲突
    """
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                # 如果键已存在于合并后的字典中，抛出异常
                raise ValueError(f"键冲突: '{key}' 在多个字典中出现。")
            merged_dict[key] = value
    return merged_dict

train_pics, valid_pics, test_pics = [], [], []
train_captions, valid_captions, test_captions = [], [], []
for pics, caps, dirs in zip([train_pics, valid_pics, test_pics], [train_captions, valid_captions, test_captions], [train_dirs, valid_dirs, test_dirs]):
    tmps = []
    for dir in dirs:
        tmps.append(sorted([os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.png')], key=extract_number))
        for x in os.listdir(dir):
            if x.endswith('.json'):
                x = os.path.join(dir, x)
                with open(x, 'r') as f:
                    data = json.load(f)
                    caps += [data]
    for each_book in tmps:
        for i in range(len(each_book)-4):
            following4 = each_book[i+1:i+5]
            pics.append([each_book[i]]+following4)

try:
    train_captions, valid_captions, test_captions = merge_dicts(train_captions), merge_dicts(valid_captions), merge_dicts(test_captions)
except ValueError as e:
    print(e)

hdf5_dir = os.path.join('..', 'oxford_data')
if not os.path.exists(hdf5_dir):
    os.makedirs(hdf5_dir)

def main():
    start_time = time.time()
    with h5py.File(os.path.join(hdf5_dir, 'oxford_hdf5'), 'w') as f:
        for subset, pics, caps in zip(['train', 'valid', 'test'], [train_pics, valid_pics, test_pics], [train_captions, valid_captions, test_captions]):
            length = len(pics)
            group = f.create_group(subset)
            images = []
            for i in range(len(pics[0])):
                images.append(group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
            text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
            for i, img_paths in enumerate(tqdm(pics, leave=True, desc="saveh5")):
                imgs = [Image.open(os.path.join(img_path)).convert('RGB') for img_path in img_paths]
                for j, img in enumerate(imgs):
                    img = np.array(img).astype(np.uint8)
                    img = cv2.imencode('.png', img)[1].tobytes()
                    img = np.frombuffer(img, np.uint8)
                    images[j][i] = img
                txt = [caps[img_path.split(os.sep)[-1]] for img_path in img_paths]
                text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])
    end_time = time.time()
    print(f'It took {round((end_time-start_time)/60)} minutes to generate hdf5 file.')

if __name__ == '__main__':
    main()