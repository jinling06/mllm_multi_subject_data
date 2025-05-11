"""
@Time: 2024/9/7 11:50
@Author: xujinlingbj
@File: prepare_m3ke.py
"""
import os
import sys

from data_process.utils import *
from pylatexenc.latex2text import LatexNodes2Text
from tqdm import tqdm


def parse_m3ke_data(data_dir, save_path):

    index = 0
    res_data = []
    map_data_path = os.path.join(data_dir, 'subject_cluster.mapping.json')
    data_dir = os.path.join(data_dir, 'data')
    mode_names = os.listdir(data_dir)
    map_data = load_json_file(map_data_path)
    file_name_2_catgory = {}

    for k, v in map_data.items():
        for x in v:
            file_name_2_catgory[x] = k
    keyword2id = get_keyword2id()
    for mode_name in mode_names:
        mode_dir = os.path.join(data_dir, mode_name)
        if not os.path.isdir(mode_dir):
            continue
        for file_name in tqdm(os.listdir(mode_dir), desc='make'):
            file_path = os.path.join(mode_dir, file_name)
            # print(file_path)
            category = file_name_2_catgory[file_name.split('.')[0]]
            data = load_jsonl_data(file_path)
            # print(f'data num={len(data)}')
            for line in data:
                question = line['question']
                answer = line['answer']
                if check_illegal_answer(answer):
                    # print(f'error answer: {answer}')
                    continue
                option_keys = ['A', 'B', 'C', 'D']
                question = LatexNodes2Text().latex_to_text(question)
                choose_text = ''
                for option in option_keys:
                    choose_text += '\n' + option +'.'+ LatexNodes2Text().latex_to_text(line[option])
                if contains_chinese(question):
                    if len(choose_text) > 0:
                        choose_text = '\n选项是:' + choose_text
                    prompt = f"根据问题描述，回答下面的问题。\n问题是： {question}。{choose_text}"
                else:
                    if len(choose_text) > 0:
                        choose_text = '\nChoices:' + choose_text
                    prompt = f"Based on the problem description, answer the following questions.\nThe question is: {question}. {choose_text}"

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
                    'id': f'm3ke_{index}',
                    'raw_id': line['id'],
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
    # 355 个
    data_dir = 'M3KE'
    save_path = 'M3KE/train_0907.json'
    parse_m3ke_data(data_dir, save_path)
