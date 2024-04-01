import os
import json
import shutil
from tqdm import tqdm

create_dir = lambda x : os.makedirs(x) if not os.path.exists(x) else None

split_result_path = os.path.join('..', '..', 'dataset', 'split')
dirs_path_list = [os.path.join(split_result_path, x) for x in os.listdir(split_result_path) if os.path.isdir(os.path.join(split_result_path, x))]


# TODO 测试部分train
valid_dirs = ['8-11', '8-08', '7-11', '7-17', '6-13', '6-20', '5-11', '5-20', '4-11', '4-24', '3-11', '3-29', '2-11', '2-25', '1-38', '1-59']
dirs_path_list = [x for x in dirs_path_list if any(y in x for y in ['1-', '2-', '3-', 'test']+valid_dirs)]
# for dirs in dirs_path_list:
    # print(dirs)
# TODO

with open('caption.json', 'r') as f:
    data = json.load(f)

newcaps_path = os.path.join('..', '..', 'dataset', 'newcaps')
create_dir(newcaps_path)
for dir in tqdm(dirs_path_list, desc='Moving', leave=True):
    image_path = [os.path.join(dir, x) for x in os.listdir(dir) if not '.json' in x] if not 'test' in dir else [os.path.join(dir, x) for x in os.listdir(dir)] 
    dir = os.path.join(newcaps_path, dir.split(os.sep)[-1])
    create_dir(dir)
    caption = {}
    for image in image_path:
        image_name = image.split(os.sep)[-1]
        shutil.copy(src=image, dst=os.path.join(dir, image_name))
        # Do not change caption in test
        if '.json' in image:
            continue
        else:
            caption[image_name] = data[image_name].replace('-', '').replace('/', '').replace('`', '').encode('utf-8').decode('unicode_escape')
    
    with open(os.path.join(dir, 'caption.json'), 'w') as f:
        json.dump(caption, f, indent=4)
