"""
@Time: 2024/8/26 18:37
@Author: xujinlingbj
@File: prepare_cmmu.py
https://huggingface.co/datasets/m-a-p/CMMMU/tree/main
"""
import os
import random
import re
import io
import sys
from tqdm import tqdm
import pandas as pd
from data_process.utils import *
import base64
from io import BytesIO
import requests
from PIL import Image
from prompt_data import global_chinese_prompt_list


def parse_cmmu_data(data_dir, save_path):
    category_dir_list = os.listdir(data_dir)
    category_dir_list = [x for x in category_dir_list if os.path.isdir(os.path.join(data_dir, x))]
    index = 0
    res_data = []
    filter_num = 0
    keyword2id = get_keyword2id()
    for category_dir_name in category_dir_list:
        category_file_path = os.path.join(data_dir, category_dir_name)
        image_save_dir = os.path.join(category_file_path, 'images')
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        data_file_list = os.listdir(category_file_path)
        for file_name in data_file_list:
            file_path = os.path.join(category_file_path, file_name)
            if os.path.isdir(file_path):
                continue
            if not file_path.endswith('.parquet'):
                continue
            print(file_path)

            data = pd.read_parquet(file_path)
            print(f'data num={len(data)}')
            for _, row in tqdm(data.iterrows(), desc='make'):

                id = row['id']
                type = row['type']
                answer = row['answer']
                if answer is None or answer == '?':
                    filter_num += 1
                    continue
                if check_illegal_answer(answer):
                    continue

                question = row['question'].replace('\n', '').strip()
                # 匹配模式：<img= 中间可以有任意字符，直到遇到 >
                pattern = r'<img=.*?>'

                # 替换为一个空格
                question = re.sub(pattern, '', question)

                options = {'option1': 'A', 'option2': 'B', 'option3': 'C', 'option4': 'D'}
                choose_text = ''
                for option, v in options.items():
                    if '<img=' not in row[option] and row[option] != '-':
                        choose_text += '\n' + str(v) + '.' + row[option]
                if choose_text != '':
                    choose_text = '选项是：' + choose_text
                image_columns = ['image_1', 'image_2', 'image_3', 'image_4']
                images = []
                for image_name in image_columns:
                    if row[image_name] is None:
                        break

                    image_file_name = row[f'{image_name}_filename']
                    image_save_path = os.path.join(image_save_dir, image_file_name)
                    if not os.path.exists(image_save_path):
                        image = Image.open(io.BytesIO(row[image_name]['bytes'])).convert("RGB")

                        image.save(image_save_path)
                    images.append(image_save_path.replace('raw_data/', ''))
                image_token = "<image>" * len(images)
                hint = "只需要回答选项字母, 从问题描述中选择答案"
                if '<img=' in row['option1']:
                    hint = "只需要回答选项字母, 从图片中选择答案"
                if '-' in row['option1']:
                    hint = "这是一个填空题，请直接给出答案"
                prompt = f"{image_token}\n根据图示，回答下面的问题。\n问题是：{question}。{choose_text}\n" f"请注意：{hint}。你的答案是："
                # answer = answer.strip()
                conversations = [
                    {
                        'from': 'human',
                        'value': prompt,
                    },
                    {
                        'from': 'gpt',
                        'value': "答案是："+answer,
                    }
                ]
                temp_dic = {
                    'id': f'cmmu_{category_dir_name}_{index}',
                    'keyword': 'none',
                    'keyword_id': keyword2id['none'],
                    'image': images,
                    'conversations': conversations,
                    'category_dir_name': category_dir_name,
                    "system": random.choice(global_chinese_prompt_list),
                }
                index += 1

                res_data.append(temp_dic)
    print(f'filter_num={filter_num}')
    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    save_path = sys.argv[1]
    parse_cmmu_data(data_dir='CMMMU',
                    save_path=save_path)
    pass