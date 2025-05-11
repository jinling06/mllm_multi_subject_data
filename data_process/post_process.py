"""
@Time: 2024/8/27 12:13
@Author: xujinlingbj
@File: post_process.py
"""
import random

from data_process.utils import load_json_file, save_json_file


def post_process(data_path, save_path):
    data = load_json_file(data_path)
    legal_ans_list = ['A', 'B', 'C', 'D']
    change_ans_num = 0
    for line in data:
        for question_info in line['example']:
            answer = question_info['model_answer'][0]
            if answer not in legal_ans_list:
                index_list = []
                for legal_ans in legal_ans_list:
                    index = answer.find(legal_ans)
                    if index != -1:
                        index_list.append([legal_ans, index])
                index_list = sorted(index_list, key=lambda x:x[1])
                if len(index_list) == 0:
                    index_list = [[random.choice(legal_ans_list), 0]]
                print('*'*10)
                print(answer)
                print(f'change to --> ', index_list[0][0])
                question_info['model_answer'] = [index_list[0][0]]
                change_ans_num += 1
    print(f'change_ans_num={change_ans_num}')
    save_json_file(save_path, data)


if __name__ == '__main__':
    post_process(data_path='output/submit_stage3-sft-math-248k.json',
                 save_path='output/submit_stage3-sft-math-248k_pos.json')
    pass