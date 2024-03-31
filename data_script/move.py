import os
import json
import shutil
from tqdm import tqdm

create_dir = lambda x : os.makedirs(x) if not os.path.exists(x) else None

split_result_path = os.path.join('..', '..', 'dataset', 'split')
dirs_path_list = [os.path.join(split_result_path, x) for x in os.listdir(split_result_path) if os.path.isdir(os.path.join(split_result_path, x))]

with open('caption.json', 'r') as f:
    data = json.load(f)

newcaps_path = os.path.join('..', '..', 'dataset', 'newcaps')
create_dir(newcaps_path)
for dir in tqdm(dirs_path_list, desc='Moving', leave=True):
    image_path = [os.path.join(dir, x) for x in os.listdir(dir) if not '.json' in x]
    dir = os.path.join(newcaps_path, dir.split(os.sep)[-1])
    create_dir(dir)
    caption = {}
    for image in image_path:
        image_name = image.split(os.sep)[-1]
        shutil.copy(src=image, dst=os.path.join(dir, image_name))
        # Do not change caption in test
        if 'test' in dir:
            if image_name in caption:
                caption[image_name] = caption[image_name]
        else:
            caption[image_name] = data[image_name]
    
    with open(os.path.join(dir, 'caption.json'), 'w') as f:
        json.dump(caption, f, indent=4)
