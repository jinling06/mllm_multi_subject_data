"""
@Time: 2024/8/27 22:25
@Author: xujinlingbj
@File: prepare_CMMLU-Clinical-Knowledge-Benchmark.py
"""
import json
import os
import sys
from tqdm import tqdm
from data_process.utils import save_json_file, load_json_file


# 纯文本数据
def parse_CMMLU_Clinical_Knowledge_Benchmark(data_path, save_path):
    data = load_json_file(data_path)
    print(f'data num ={len(data)}')
    res_data = []
    index = 0
    for line in tqdm(data, desc='make'):
        question = line['question']
        choose_list = ['A', 'B', 'C', 'D']
        option_text = ''
        for x in choose_list:
            option_text += '\n'+x + '、' + line[x]
        answer = line['label']
        if len(answer) > 10:
            continue
        prompt = f'这是一个纯文本选择题，请根据问题描述，选出最佳选项。\n问题是：{question}\n选项是：{option_text}'
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
            'id': f'CMMU_CKB_{index}',
            'conversations': conversations,
        }
        index += 1

        res_data.append(temp_dic)

    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == '__main__':
    # 医学纯文本题目
    data_path = 'CMMLU-Clinical-Knowledge-Benchmark/clinical_knowledge.json'
    save_path = 'CMMLU-Clinical-Knowledge-Benchmark/train_0827.json'
    parse_CMMLU_Clinical_Knowledge_Benchmark(data_path, save_path)
    pass