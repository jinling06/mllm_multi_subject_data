"""
@Time: 2024/9/8 09:11
@Author: xujinlingbj
@File: prepare_mmlu.py
"""
import os
import sys

import pandas as pd
from tqdm import tqdm
from pylatexenc.latex2text import LatexNodes2Text
from data_process.utils import *


def parse_mmlu(data_dir, save_path):
    res_data = []
    index = 0
    saved_question = []
    legal_subject_list = ['college_biology', 'college_chemistry', 'college_mathematics', 'college_physics',
                           'conceptual_physics', 'high_school_biology', 'high_school_chemistry', 'high_school_geography',
                           'high_school_mathematics', 'high_school_physics', 'high_school_us_history', 'high_school_world_history',

                           ]
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            print(f'filter {category_dir}')
            continue

        # if category not in ['all']:
        #     continue
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            if not file_path.endswith('.parquet'):
                continue
            print(file_path)

            data = pd.read_parquet(file_path)
            for idd, row in tqdm(data.iterrows(), total=len(data), desc='make'):
                if 'train' in row:
                    row = row['train']
                question = row['question']
                if question in saved_question:
                    print(f'question have saved:{question}')
                    continue
                saved_question.append(question)
                subject = row['subject']
                if subject not in legal_subject_list:
                    continue
                choices = row['choices']
                answer = row['answer']
                question = LatexNodes2Text().latex_to_text(str(question))
                choose_text = ''
                st_a = 'A'
                for x in choices:
                    x = LatexNodes2Text().latex_to_text(str(x))
                    choose_text += '\n' + st_a + '、' + x
                    st_a = chr(ord(st_a) + 1)
                if len(choose_text) > 0:
                    choose_text = '\nChoices:' + choose_text
                prompt = f"Based on the problem description, answer the following questions.\nThe question is: {question} {choose_text}"
                prompt = re.sub(r'(Statement 1)', r'\n\1', prompt)
                prompt = re.sub(r'(Statement 2)', r'\n\1', prompt)
                option_id_list = ['A', 'B', 'C', 'D']
                answer = option_id_list[int(answer)]
                if check_illegal_answer(answer):
                    print(f'answer error:{answer}')
                    continue
                conversations = [
                    {
                        'from': 'human',
                        'value': prompt,
                    },
                    {
                        'from': 'gpt',
                        'value': "答案是：" + answer,
                    }
                ]
                temp_dic = {
                    'id': f'mmlu_{index}',
                    'category': category,
                    'subject': subject,
                    'keyword': 'none',
                    'keyword_id': keyword2id['none'],
                    'conversations': conversations,
                }
                index += 1

                res_data.append(temp_dic)
    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    # 选项和问题都很长,有英文阅读理解的题目 114k
    # 只选特定类别 2510
    data_dir = 'mmlu'
    save_path = 'mmlu/train_0907.json'
    parse_mmlu(data_dir, save_path)
    pass
