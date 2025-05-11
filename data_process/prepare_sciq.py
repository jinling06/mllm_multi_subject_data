"""
@Time: 2024/9/7 20:50
@Author: xujinlingbj
@File: prepare_sciq.py
"""
import os
import random
import sys

import pandas as pd
from tqdm import tqdm

from data_process.utils import *


def parse_sciq(data_dir, save_path):
    res_data = []
    index = 0
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if not file_path.endswith('.parquet'):
            continue
        print(file_path)
        data = pd.read_parquet(file_path)
        print(f'data num={len(data)}')
        for idd, row in tqdm(data.iterrows(), desc='make'):

            question = row['question']
            distractor3 = row['distractor3']
            distractor1 = row['distractor1']
            distractor2 = row['distractor2']
            correct_answer = row['correct_answer']
            support = row['support']

            options = [distractor1, distractor2, distractor3, correct_answer]
            random.shuffle(options)
            st_a = 'A'
            choose_text = ''
            answer = ''
            for x in options:
                choose_text += '\n' + st_a + '、' + x
                if x == correct_answer:
                    answer = st_a
                st_a = chr(ord(st_a) + 1)

            if choose_text != '':
                choose_text = 'Choices:' + choose_text
            prompt = f"Based on the problem description, answer the following questions.\nThe question is: {question}。{choose_text}\n"
            if check_illegal_answer(answer):
                print(f'error answer: {answer}')
                continue
            conversations = [
                {
                    'from': 'human',
                    'value': prompt,
                },
                {
                    'from': 'gpt',
                    'value': "The answer is: " + answer,
                }
            ]
            temp_dic = {
                'id': f'sciq_{index}',
                'keyword': 'none',
                'keyword_id': keyword2id['none'],
                'conversations': conversations,
                'support': support,

            }
            index += 1

            res_data.append(temp_dic)

    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    # 13k
    data_dir = 'sciq/data'
    save_path = 'sciq/train_0907.json'
    parse_sciq(data_dir, save_path)
    pass