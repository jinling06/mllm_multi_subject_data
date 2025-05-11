"""
@Time: 2024/8/27 21:51
@Author: xujinlingbj
@File: prepare_mathverse.py
"""
import json
import os
import sys
from tqdm import tqdm
from data_process.utils import *


# 题目在图片中
def parse_mathverse(data_path, save_path):
    data = load_json_file(data_path)
    print(f'data num ={len(data)}')
    res_data = []
    index = 0
    question_type_list = []
    for line in tqdm(data, desc='parse'):
        sample_index = line['sample_index']
        problem_index = line['problem_index']
        question_type = line['question_type']
        if question_type not in question_type_list:
            question_type_list.append(question_type)

        prompt = line['query_wo']
        answer = line['answer']

        if answer not in ['A', 'B', 'C', 'D']:
            continue
        image_token = "<image>\n"
        image = line['image']
        assert image != ''
        image = os.path.join('llava_sft/math_data/MathVerse/images', image)
        save_image_path = os.path.join('raw_data/', image)
        if not os.path.exists(save_image_path):
            print(f'image not find: {save_image_path}')
            continue
        if check_illegal_answer(answer):
            continue
        if '\nQuestion:' not in prompt:
            continue
        prompt = image_token + prompt
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
            'id': f'mathverse_{sample_index}_{problem_index}_{index}',

            'image': image,
            'conversations': conversations,

        }
        index += 1

        res_data.append(temp_dic)
    print(f'question_type_list={question_type_list}')
    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    # 1676
    data_path = 'MathVerse/testmini.json'
    save_path = sys.argv[1]
    parse_mathverse(data_path, save_path)
    pass