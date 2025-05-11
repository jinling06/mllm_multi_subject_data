"""
@Time: 2024/9/8 10:32
@Author: xujinlingbj
@File: prepare_mmcu.py
"""
import os
import sys

import pandas as pd
from tqdm import tqdm
from data_process.utils import *


def parse_mmcu(data_dir, save_path):
    res_data = []
    index = 0
    legal_subject_list = [
        '教育_化学', '教育_历史', '教育_地理', '教育_政治', '教育_数学', '教育_物理', '教育_生物', '教育_语文'
    ]
    for mode_name in os.listdir(data_dir):
        mode_dir = os.path.join(data_dir, mode_name)
        if not os.path.isdir(mode_dir):
            continue
        for file_name in os.listdir(mode_dir):
            file_path = os.path.join(mode_dir, file_name)
            category = file_name.split('.')[0]
            if not file_path.endswith('.xlsx'):
                continue
            if category not in legal_subject_list:
                continue

            data = pd.read_excel(file_path)
            for idd, row in tqdm(data.iterrows(), total=len(data), desc='make'):
                question = row['题目']
                # option_list = ['选项A', '选项B', '选项C', '选项D']
                option_list = list(row.keys())[1:5]

                try:
                    answer = row['正确答案']
                except:
                    answer = row['B']
                choose_text = ''
                st_a = 'A'
                for option_id in option_list:

                    option = row[option_id]
                    option = str(option)

                    # 使用 re.sub 进行替换，只替换第一次出现的匹配
                    option = re.sub(r'[A-Z].', ' ', option, count=1)
                    if not contains_uppercase_abcd(option_id):
                        option_id = st_a
                        st_a = chr(ord(st_a) + 1)
                    option = option.strip()
                    choose_text += '\n' + option_id.replace('选项', '') + '、' + str(option)
                if check_illegal_answer(answer):
                    print(f'answer error:{answer}')
                    continue
                if len(choose_text) > 0:
                    choose_text = '\n选项是:' + choose_text
                prompt = f"根据问题描述，回答下面的问题。\n问题是：{question}。{choose_text}"
                answer = answer.strip()
                answer_num = compute_answer_num(answer)
                if answer_num == 0:
                    print(f'illegal  error:{answer}')
                    continue
                if answer_num > 1:
                    answer = format_answer(answer)
                    if len(answer.split('、')) != answer_num:
                        print(f'answer  error:{answer}')
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
                    'id': f'mmcu_{index}',
                    'keyword': 'none',
                    'keyword_id': keyword2id['none'],
                    'category': category,
                    'conversations': conversations,
                }
                index += 1

                res_data.append(temp_dic)
    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    # 大部分是填空的选择题 9303
    # 只选特定科目 3342
    data_dir = 'MMCU/MMCU0513'
    save_path = 'MMCU/train_0907.json'
    parse_mmcu(data_dir, save_path)
    pass