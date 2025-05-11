"""
@Time: 2024/8/26 12:07
@Author: xujinlingbj
@File: prepare_mathv360k.py
"""
import json
import os
import random
import sys
from tqdm import tqdm

from data_process.prompt_data import english_prompt_list
from data_process.utils import *

def parse_zh(data_path_list, save_path):
    res_data = []
    for data_path in data_path_list:
        print(data_path)
        data = load_jsonl_data(data_path)
        for line in data:
            conversations = line['conversations']
            answer = conversations[1]['value']
            answer = "答案是: " + answer
            if len(answer) > 10:
                continue
            conversations[1]['value'] = answer
            res_data.append(line)
    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


def process_data(data_path, save_path):
    data = load_json_file(data_path)
    print(f'data num ={len(data)}')
    res_data = []
    index = 0
    keyword2id = get_keyword2id()
    for line in tqdm(data, desc='mathv360k'):
        id = line['id']
        image = line['image']
        conversations = line['conversations']
        image = os.path.join('llava_sft/math_data/MathV360K/data_images', image)
        keyword = image.split('/')[-3]

        save_image_path = os.path.join('raw_data/', image)
        if not os.path.exists(save_image_path):
            print(f'image not find: {save_image_path}')
            continue
        # conversations[0]['value'] = conversations[0]['value'] + '\nYour answer is:'
        # conversations[1]['value'] = conversations[1]['value'].replace('The answer is', '').strip()
        conversations[0]['value'] = conversations[0]['value'] + '\ntell me your answer.'
        answer = conversations[1]['value'].replace('The answer is', '')
        conversations[1]['value'] = 'The answer is: ' + answer
        if len(conversations[1]['value']) > 20:
            continue
        # answer = answer.strip()
        # answer = 'The answer is: ' + answer
        # conversations[1]['value'] = answer
        assert len(conversations) == 2
        temp_dic = {
            'id': f'mathv360k_{index}',
            'raw_id': id,
            'image': image,
            'keyword': 'none',
            'keyword_id': keyword2id['none'],
            'conversations': conversations,
            "system": random.choice(english_prompt_list),
        }
        index += 1

        res_data.append(temp_dic)
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    save_path = sys.argv[1]
    process_data(data_path='MathV360K/train_samples_all_tuning.json',
                 save_path=save_path)
