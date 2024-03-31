import os
import re
import cv2
import h5py
import json
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

split_result_path = os.path.join('..', '..', 'dataset', 'newcaps')
dirs_path_list = [os.path.join(split_result_path, x) for x in os.listdir(split_result_path) if os.path.isdir(os.path.join(split_result_path, x))]

# 自己构建一个新的test文件夹做test
test_dirs = [x for x in dirs_path_list if 'test' in x]
assert len(test_dirs) == 1
# 8-11, 8-08, 7-11, 7-17, 6-13, 6-20, 5-11, 5-20, 4-11, 4-24, 3-11, 3-29, 2-11, 2-25, 1-38, 1-59做valid
valid_dirs = [x for x in dirs_path_list if any(y in x for y in ['8-11', '8-08', '7-11', '7-17', '6-13', '6-20', '5-11', '5-20', '4-11', '4-24', '3-11', '3-29', '2-11', '2-25', '1-38', '1-59'])]
assert len(valid_dirs) == 16
# 其他做train
train_dirs = [x for x in dirs_path_list if not x in test_dirs and not x in valid_dirs]
assert len(train_dirs) == len(dirs_path_list)-len(test_dirs)-len(valid_dirs)

def extract_number(s)->int:
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
    png_list = []
    for dir in dirs:
        # 加载png图片(注意一定要排序) 双重列表: 内层列表为一本书下的所有图片
        png_list.append(sorted([os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.png')], key=extract_number))
        # 加载caption文本
        for x in os.listdir(dir):
            if x.endswith('.json'):
                x = os.path.join(dir, x)
                with open(x, 'r') as f:
                    data = json.load(f)
                    caps += [data]
    
    for each_book in png_list:
        for i in range(len(each_book)-4):
            following4 = each_book[i+1:i+5]
            pics.append([each_book[i]]+following4)

merge_test_dict = merge_dicts(test_captions)
with open(os.path.join('..', '..', 'test_text.json'), 'w', encoding='utf-8') as f:
     json.dump(merge_test_dict, f, indent=4, ensure_ascii=False)

try:
    train_captions, valid_captions, test_captions = merge_dicts(train_captions), merge_dicts(valid_captions), merge_dicts(test_captions)
except ValueError as e:
    print(e)

# 每个items为语义连续的5张图片
print(f'{len(train_pics)} items in train set, {len(valid_pics)} items in valid set, {len(test_pics)} items in test set')

hdf5_dir = os.path.join('..', '..', 'oxford_data')
hdf5_path = os.path.join(hdf5_dir, 'oxford.hdf5')
if not os.path.exists(hdf5_dir):
    os.makedirs(hdf5_dir)

def main():
    start_time = time.time()
    with h5py.File(hdf5_path, 'w') as f:
        for subset, pics, caps in zip(['train', 'val', 'test'], [train_pics, valid_pics, test_pics], [train_captions, valid_captions, test_captions]):
            length = len(pics)
            group = f.create_group(subset)
            images = []
            for i in range(len(pics[0])):
                images.append(group.create_dataset(f'image{i}', (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
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

def check_hdf5(file : str)->bool:
    if not os.path.exists(file):
        return False

    with h5py.File(file, 'r') as f:
        for subset in f:
            print(f'subet = {subset}')
            for group in f[subset]:
                print(f'group = {group}')

    return True

if __name__ == '__main__':
    if not check_hdf5(file=hdf5_path):
        main()