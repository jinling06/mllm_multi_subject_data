"""
@Time: 2024/8/27 22:06
@Author: xujinlingbj
@File: prepare_MathVista.py
"""
import os
import sys

import pandas as pd
from tqdm import tqdm
from data_process.utils import *


def parse_mathvista(data_dir, save_path):
    file_names = os.listdir(data_dir)
    index = 0
    res_data = []
    question_list = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not file_path.endswith('.parquet'):
            continue
        print(file_path)
        data = pd.read_parquet(file_path)
        print(f'data num={len(data)}')
        for _, row in tqdm(data.iterrows(), total=len(data), desc='make'):
            idd = row['pid']
            question = row['question']
            image = row['image']
            choices = row['choices']
            answer = row['answer']
            query = row['query']

            if question in question_list:
                continue
            question_list.append(question)
            image = os.path.join('llava_sft/math_data/MathVista', image)
            save_image_path = os.path.join('raw_data/', image)
            if not os.path.exists(save_image_path):
                print(f'image not find={save_image_path}')
                continue
            choose_text = ''
            change_ans = False
            st_a = 'A'
            if choices is not None:
                for x in choices:
                    if change_ans is False and answer == x:
                        answer = st_a
                        change_ans = True
                    choose_text += '\n' + st_a + '、' + x
                    st_a = chr(ord(st_a) + 1)
            if choose_text != '':
                choose_text = 'Choices:' + choose_text
            hint = "This is a multiple-choice question, choose the answer from the question description"
            if choices is None:
                hint = "This is a quiz question, please give your answer directly"
            image_token = "<image>"
            prompt = f"{image_token}\nBased on the diagram, answer the following questions.\nThe question is: {question}。{choose_text}\n" f"Please note:{hint}."
            if check_illegal_answer(answer):
                continue
            if question == '':
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
                'id': f'mathvista_{idd}_{index}',

                'image': image,
                'conversations': conversations,

            }
            index += 1

            res_data.append(temp_dic)

    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    # 461
    data_dir = 'MathVista/data'
    save_path = sys.argv[1]
    parse_mathvista(data_dir, save_path)
    pass