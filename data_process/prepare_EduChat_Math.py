"""
@Time: 2024/11/9 16:08
@Author: xujinlingbj
@File: prepare_EduChat_Math.py
"""
import os.path
import sys
from pylatexenc.latex2text import LatexNodes2Text
import pandas as pd

from data_process.utils import *

"""
1、各年级分布：{'高二': 3158, '七年级': 2209, '九年级': 2393, '六年级': 2246, '八年级': 1733, '四年级': 2784, '二年级': 2087, '高三': 2492, '高一': 2871, '一年级': 1499, '三年级': 2124, '五年级': 2473}
2、学科分布：{'解析几何': 3463, '度量几何学': 3614, '算术': 9854, '代数': 3316, '计数': 1953, '变换几何': 359, '图论': 70, '组合几何学': 635, '立体几何学': 2638, '组合数学': 964, '逻辑题': 543, '画法几何学': 600, '统计数学': 60}
3、是否带图分布：{'no': 20119, 'yes': 7950}
4、题型分布：选择题列表 10840 填空题列表 8050 解答题列表 9083 判断题列表 106

这个数据原本只有2万8， 带图的只有将近8k， 带图且是选择题的只有2927, 符合我们任务的数据有点少
"""


def reformat_options(text):
    text = replace_last_str(text, 'A．', 'A、')
    text = replace_last_str(text, 'A.', 'A、')
    text = replace_last_str(text, 'B．', 'B、')
    text = replace_last_str(text, 'B.', 'B、')
    text = replace_last_str(text, 'C．', 'C、')
    text = replace_last_str(text, 'C.', 'C、')
    text = replace_last_str(text, 'D．', 'D、')
    text = replace_last_str(text, 'D.', 'D、')
    text = text.replace('<ImageHere>', '')
    text = text.replace('\nA、\nB、\nC、\nD、', '\nA、A\nB、B\nC、C\nD、D')
    return text


def reformat_prompt(text):
    while ' ' in text:
        text = text.replace(" ", "")
    while '\n\n' in text:
        text = text.replace("\n\n", "\n")
    return text


def change_to_train_format(data, image_dir):
    res_data = []
    for line in tqdm(data, total=len(data), desc='format'):
        image_list = line['image']
        question = line['question']
        idd = line['level']
        index = line['id']
        options = line['options']
        answer = line['answer']
        image_num = question.count('<ImageHere>') + options.count('<ImageHere>')
        image_list = image_list[:image_num]
        if len(image_list) > 4:
            print(f'过滤图片个数大于4的数据: {image_list}')
            continue
        question = LatexNodes2Text().latex_to_text(question)
        options = LatexNodes2Text().latex_to_text(options)
        # options = options.split('\n')
        # options = [x for x in options if len(x) > 0]
        # if len(options) < 4:
        #     print(f'选项错误：{line}')
        #     continue

        for i in range(len(image_list)):
            image_path = image_list[i]

            image_list[i] = os.path.join(image_dir, image_path)
            if not os.path.exists(image_list[i]):
                print(f'miss image:{image_list[i]}')
                sys.exit(0)
            image_list[i] = image_list[i].replace('raw_data/', '')
        if check_illegal_answer(answer):
            print(f'不合法答案：{answer}')
            continue
        options = reformat_options(options)

        image_token = ''
        if len(image_list) > 0:
            image_token = "<image>" * len(image_list)
            image_token += '\n'

        prompt = f"{image_token}根据问题描述，回答下面的问题。\n问题是：{question}\n选项是：\n{options}\n你的答案是："

        prompt = prompt.replace('<ImageHere>', '')
        prompt = reformat_prompt(prompt)
        conversations = [
            {
                "from": "human",
                "value": prompt,
            },
            {
                "from": "gpt",
                "value": "答案是：" + answer,
            },
        ]
        temp_dic = {
            "id": f"edu_chat_math_{idd}_{index}",
            "keyword": "math",
            "conversations": conversations,
            "raw_answer": line['answer'],
        }
        if len(image_list) > 0:
            temp_dic["image"] = image_list
        res_data.append(temp_dic)
    return res_data


def parse_edu_chat_math(data_dir, image_dir, save_path):
    jsonl_data_path = os.path.join(data_dir, 'all_data.jsonl')
    question_type_data_path = os.path.join(data_dir, 'Question_type_index.txt')
    jsonl_data = load_jsonl_data(jsonl_data_path)
    print(f'raw data num={len(jsonl_data)}')
    # ['选择题', '选择题列表', '填空题', '填空题列表', '解答题', '解答题列表', '判断题', '判断题列表']
    question_type_data = load_json_file(question_type_data_path)
    print(question_type_data.keys())
    question_id_2_type = defaultdict(str)
    for key, value in question_type_data.items():
        if not isinstance(value, list):
            continue
        for x in value:
            assert x not in question_id_2_type
            question_id_2_type[x] = key

    all_level = defaultdict(int)
    have_image_info = []
    subject_info = defaultdict(int)
    cnt = 0
    res_data = []
    all_image_num = 0
    for line in tqdm(jsonl_data):
        level = line['level']
        all_level[level] += 1
        subject = line['subject']
        subject_info[subject] += 1
        image_list = line['image']
        question = line['question']

        options = line['options']
        image_num = question.count('<ImageHere>') + options.count('<ImageHere>')

        all_image_num += len(line['image'])
        question_id = line['id']
        if level in ['高二', '高三', '高一'] and question_id_2_type[question_id] in ['选择题列表']:

            res_data.append(line)
            have_image_info.append([image_num])
    print(f'all_level={all_level}')
    have_image_info = pd.DataFrame(have_image_info, columns=['num'])
    print(have_image_info['num'].describe(percentiles=[.99, .95, .9, .8, .7, .6, .5, .4, .3]))
    print(f'all_image_num={all_image_num}')
    print(f'subject_info={subject_info}')
    print(f'res_data num: {len(res_data)}')

    res_data = change_to_train_format(res_data, image_dir)
    print(f'format data num ={len(res_data)}')
    save_json_file(save_path, res_data)


if __name__ == '__main__':

    data_dir = 'EduChat-Math/data'
    image_dir = 'EduChat-Math/Images/All_Images'
    save_path = f'{data_dir}/train_1109.json'
    parse_edu_chat_math(data_dir, image_dir, save_path)
    pass