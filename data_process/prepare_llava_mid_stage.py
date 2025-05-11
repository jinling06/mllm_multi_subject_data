"""
@Time: 2024/8/29 22:06
@Author: xujinlingbj
@File: prepare_llava_mid_stage.py
"""
import os
import sys
from tqdm import tqdm
from data_process.utils import *


def parse_llava_mid_stage(data_dir, save_path):
    category_dirs = os.listdir(data_dir)
    res_data = []
    index = 0
    for category_name in category_dirs:
        category_dir = os.path.join(data_dir, category_name)
        if not os.path.isdir(category_dir):
            continue
        files = os.listdir(category_dir)
        for file_name in files:
            if '.json' not in file_name:
                continue
            file_path = os.path.join(category_dir, file_name)
            print(file_path)
            data = load_json_file(file_path)
            print(f'data num={len(data)}')
            for line in tqdm(data, desc=f'{file_name}'):
                if 'image' in line:
                    image = line['image']
                    image = os.path.join('LLaVA-OneVision-Mid-Data', category_name, image)
                    save_image_path = os.path.join('raw_data/', image)
                    if not os.path.exists(save_image_path):
                        print(f'image not find: {save_image_path}')

                        continue

                    line['image'] = image
                if len(line['conversations'][0]['value']) == 0:
                    print(f'prompt error: {line["conversations"][0]["value"]}')
                    continue
                if len(line['conversations'][1]['value']) == 0:
                    print(f'ans error: {line["conversations"][1]["value"]}')
                    continue
                line['raw_id'] = line['id']
                line['id'] = f'llava_mid_stage_{index}'
                index += 1
                res_data.append(line)
    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    data_dir = 'LLaVA-OneVision-Mid-Data'
    save_path = 'LLaVA-OneVision-Mid-Data/train_mid_stage_0829.json'
    parse_llava_mid_stage(data_dir, save_path)
    pass
