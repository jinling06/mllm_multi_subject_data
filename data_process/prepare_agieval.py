"""
@Time: 2024/9/8 09:56
@Author: xujinlingbj
@File: prepare_agieval.py
"""
import os
import re
import sys

from tqdm import tqdm
from pylatexenc.latex2text import LatexNodes2Text
from data_process.utils import *


# 定义一个函数来处理每个选项字符串
def process_option(option):
    # 提取第一个大写字母及其所在的小括号或后面的特定字符
    uppercase_letter_match = re.search(r'\([A-Z]\)|[A-Z][:、.]', option)
    if uppercase_letter_match:
        uppercase_letter = uppercase_letter_match.group(0)
    else:
        uppercase_letter = ''

    # 只去掉第一个大写字母及其所在小括号或后面的特定字符
    cleaned_option = re.sub(r'\([A-Z]\)|[A-Z][:、.]', '', option, count=1).strip()

    return uppercase_letter, cleaned_option


def parse_agieval(data_dir, save_path):
    res_data = []
    index = 0
    legal_subject_list = [
        'gaokao-biology', 'gaokao-chemistry', 'gaokao-chinese', 'gaokao-geography', 'gaokao-history',
        'gaokao-mathqa', 'gaokao-physics'
    ]
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not file_path.endswith('.jsonl'):
            print(f'filter file={file_path}')
            continue
        category = file_name.split('.')[0]
        if category not in legal_subject_list:
            continue
        print(file_path)
        data = load_jsonl_data(file_path)
        for line in tqdm(data, total=len(data), desc='make'):
            passage = line['passage']
            if not passage:
                passage = ''
            question = line['question']
            options = line['options']
            answer = line['label']
            explanation = line['other']
            question = LatexNodes2Text().latex_to_text(str(question))
            choose_text = ''
            if options is None:
                # print(f'line error ：{options}')

                continue
            # 处理每个选项
            for option in options:
                # letters, cleaned = process_option(option)
                # if len(letters) == 0:
                #     raise ValueError(f'选项解析错误：{letters}, option={options}')
                # letters = letters[0]
                cleaned = LatexNodes2Text().latex_to_text(option)
                choose_text += '\n' + cleaned
            if len(passage) > 0:
                passage += '\n'
            if contains_chinese(question):
                if len(choose_text) > 0:
                    choose_text = '\n选项是:' + choose_text
                prompt = f"根据问题描述，回答下面的问题。\n问题是：{passage}{question}。{choose_text}"
            else:
                if len(choose_text) > 0:
                    choose_text = '\nChoices:' + choose_text
                prompt = f"Based on the problem description, answer the following questions.\nThe question is: {question}. {choose_text}"
            if isinstance(answer, list):
                if len(answer) > 1:
                    print(f'multi answer: {answer}')
                    continue
                answer = answer[0]
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
                'id': f'agieval_{index}',
                'keyword': 'none',
                'category': category,
                'keyword_id': keyword2id['none'],
                'conversations': conversations,
            }
            index += 1

            res_data.append(temp_dic)
    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    # 6165 只保留特定科目 1639
    data_dir = 'AGIEval/data/v1_1'
    save_path = 'AGIEval/train_0907.json'
    parse_agieval(data_dir, save_path)
    pass