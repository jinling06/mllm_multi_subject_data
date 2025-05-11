"""
@Time: 2024/10/30 10:23
@Author: xujinlingbj
@File: check_format.py
"""
import json
import sys

from data_process.utils import *


def check_common_info(data_path):
    data = load_json_file(data_path)
    print(len(data))
    # data = [x for x in data if x['common_info']['common_ratio_base_custom'] > 0.3 or x['common_info']['common_ratio_base_test'] > 0.3]
    data = [x for x in data if
            x['common_info']['intersection_and_union'] > 0.4]

    sample_data = random.choices(data, k=10)
    for line in sample_data:
        try:
            prompt = line['conversations'][0]['value']
        except:
            prompt = line['prompt']
        common_text = line['common_info']
        common_ratio = line['common_info']

        print('+' * 30)
        print(prompt)
        print(json.dumps(line['common_info'], ensure_ascii=False, indent=4))


def check_answer(data_path):
    data = load_json_file(data_path)
    print(len(data))
    sample_data = random.choices(data, k=10)
    for line in sample_data:
        prompt = line['conversations'][0]['value']
        answer = line['conversations'][1]['value']
        raw_answer = line.get('raw_answer', '')
        print('+' * 30)
        print(prompt)
        print(answer)
        print('==')
        print(raw_answer)


def check_clean_data_format(data_path):
    data = load_json_file(data_path)
    print(len(data))
    sample_data = random.choices(data, k=10)
    for line in sample_data:

        try:
            prompt = line['conversations'][0]['value']
        except:
            prompt = line['prompt']
        print('+' * 30)
        print(prompt)


if __name__ == '__main__':

    data_path = '/mnt/data/xujinlingbj/raw_data/llava_sft/stage2_train_1102_filter.json'
    # 0 answer 1 format 2 common_info
    check_type = 0
    if check_type == 0:
        check_answer(data_path)
    elif check_type == 1:
        check_clean_data_format(data_path)
    elif check_type == 2:
        check_common_info(data_path)
