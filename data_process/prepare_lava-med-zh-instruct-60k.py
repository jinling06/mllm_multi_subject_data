"""
@Time: 2024/8/26 19:41
@Author: xujinlingbj
@File: prepare_lava-med-zh-instruct-60k.py
"""
import io
import os
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

from data_process.utils import save_json_file


def parse_lava_med_zh_instruct_60k(data_dir, save_path):
    """不是选择题"""
    file_names = os.listdir(data_dir)
    image_save_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    res_data = []
    index = 0
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not file_path.endswith('.parquet'):
            continue
        print(file_path)
        data = pd.read_parquet(file_path)
        print(f'data num={len(data)}')
        for _, row in tqdm(data.iterrows(), total=len(data), desc='make'):
            messages = row['messages']
            images_info = row['images']
            conversations = []
            for i in range(0, len(messages), 2):
                conversations.append({'from': 'human',
                        'value': messages[i]['content']})
                conversations.append({'from': 'gpt',
                                      'value': messages[i+1]['content']})
            images = []
            for one_image_info in images_info:
                image_file_name = one_image_info['path']

                image_save_path = os.path.join(image_save_dir, image_file_name)
                if not os.path.exists(image_save_path):
                    image = Image.open(io.BytesIO(one_image_info['bytes'])).convert("RGB")

                    image.save(image_save_path)
                images.append(image_save_path.replace('raw_data/', ''))
            image_token = "<image>" * len(images)
            conversations[0]['value'] = image_token + '\n' + conversations[0]['value']
            temp_dic = {
                'id': f'lava_med_zh_instruct_60k_{index}',
                'images': images,
                'conversations': conversations,

            }
            index += 1

            res_data.append(temp_dic)

    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)


if __name__ == '__main__':
    parse_lava_med_zh_instruct_60k(data_dir='llava-med-zh-instruct-60k/data',
                                   save_path='llava-med-zh-instruct-60k/llava_med_zh_instruct_60k_train.json')