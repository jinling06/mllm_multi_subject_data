"""
@Time: 2024/9/8 09:24
@Author: xujinlingbj
@File: preprare_cmmlu.py
"""
import os
import sys
from pylatexenc.latex2text import LatexNodes2Text
import pandas as pd
from tqdm import tqdm

from data_process.utils import *


def parse_cmmlu_data(data_dir, save_path):
    res_data = []
    index = 0
    for mode_name in os.listdir(data_dir):
        mode_dir = os.path.join(data_dir, mode_name)
        if not os.path.isdir(mode_dir):
            continue
        for file_name in os.listdir(mode_dir):
            file_path = os.path.join(mode_dir, file_name)
            category = file_name.split('.')[0]
            if not file_path.endswith('.csv'):
                # print(f'error file={file_path}')
                continue
            data = pd.read_csv(file_path)
            for idd, row in tqdm(data.iterrows(), desc='make'):

                question = row['Question']
                question = LatexNodes2Text().latex_to_text(question)
                option_id_list = ['A', 'B', 'C', 'D']
                answer = row['Answer']
                choose_text = ''
                for option_id in option_id_list:
                    option = LatexNodes2Text().latex_to_text(str(row[option_id]))
                    choose_text += '\n'+str(option_id) +'.' + option
                if choose_text != '':
                    choose_text = '选项是：' + choose_text
                prompt = f"基于问题描述，回答下面的问题。\n问题是： {question}。{choose_text}\n"
                if check_illegal_answer(answer):
                    # print(f'error answer: {answer}')
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
                    'id': f'cmmlu_{index}',
                    'category': category,
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
    # 11k
    data_dir = 'cmmlu'
    save_path = 'cmmlu/train_0907.json'
    parse_cmmlu_data(data_dir, save_path)
    pass