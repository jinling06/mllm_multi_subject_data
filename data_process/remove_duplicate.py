"""
@Time: 2024/9/13 15:44
@Author: xujinlingbj
@File: remove_duplicate.py
"""
import os
import re
import shutil
import sys
from pylatexenc.latex2text import LatexNodes2Text
import concurrent.futures

import pandas as pd
from tqdm import tqdm

from data_process.utils import *


def get_question_from_conv(prompt):
    index = prompt.find('回答下面的问题。')
    if index == -1:
        index = 0
    prompt = prompt[index:].replace('<image>', '').replace('回答下面的问题。', '').replace('你的答案是：', '')
    prompt = prompt.replace('问题是：', '').replace('\n', '').strip()
    return prompt


def filter_same_question_from_file(data, base_data):
    """去除data中和base data重复的数据"""
    saved_question = []
    res_data = []

    for line in tqdm(base_data, total=len(base_data), desc='same'):
        if 'question_url' not in line and 'source_id' not in line:

            continue
        if 'question_url' in line:
            question_url = line['question_url']
        else:
            question_url = line['source_id'].split('_')[0]

        saved_question.append(question_url)
    for line in tqdm(data, total=len(data), desc='same'):
        if 'question_url' not in line and 'source_id' not in line:
            continue
        if 'question_url' in line:
            question_url = line['question_url']
        else:
            question_url = line['source_id'].split('_')[0]
        if question_url in saved_question:
            continue
        res_data.append(line)

    print(f'过滤相同题目后，剩余数量={len(res_data)}')
    return res_data


def filter_same_question(data):
    saved_question = []
    res_data = []

    for line in tqdm(data, total=len(data), desc='same'):
        if 'question_url' not in line and 'source_id' not in line:
            print(f'无 question_url 和 source_id，不进行去重')
            return data
        if 'question_url' in line:
            question_url = line['question_url']
        else:
            question_url = line['source_id']

        if question_url in saved_question:
            continue

        saved_question.append(question_url)
        res_data.append(line)
    print(f'过滤相同题目后，剩余数量={len(res_data)}')
    return res_data


def parse_thread_data(filter_data, test_data):
    res_data = []
    # 和测试集去重
    for line in tqdm(filter_data, total=len(filter_data), desc='filter'):

        keyword = line['keyword']
        question = line['question']

        common_info = remove_duplicate_with_intersection_and_union_in_file(test_data, question, set(question),
                                                                           min_ratio=0.5)
        line['common_info'] = common_info

        res_data.append(line)
    return res_data


def filter_in_file_data(filter_data):
    res_data = []
    filter_data = sorted(filter_data, key=lambda x: len(x['question']))

    question_set_list = [{'question_set': set(x['question']), 'question': x['question']} for x in filter_data]
    max_idx = []
    data_len = [len(x['question']) for x in filter_data]

    # 文件内部去重
    for i, line in tqdm(enumerate(filter_data), total=len(filter_data), desc='filter'):
        # keyword = line['keyword']
        question = line['question']
        now_question_set = question_set_list[i]['question_set']
        # retain_data = [x for x in question_set_list[i+1:] if len(x['question']) / len(question) < 2]
        end_index = find_first_greater_than(data_len[i+1:], data_len[i]*2)
        retain_data = question_set_list[i+1:]
        retain_data = retain_data[:end_index]

        common_info = remove_duplicate_with_intersection_and_union_in_file(retain_data, question, now_question_set,
                                                                           min_ratio=0.8)
        # if common_info['intersection_and_union'] == 0:

        line['common_info_in_file'] = common_info

        res_data.append(line)
    return res_data


def get_thread_group_data(data):
    res_data = []
    for line in tqdm(data, total=len(data), desc='get_group_data'):
        keyword = line['keyword'].lower()
        prompt = line['conversations'][0]['value']
        prompt = get_question_from_conv(prompt)
        prompt = prompt.replace('%', '\\%')
        latex_question = LatexNodes2Text().latex_to_text(prompt)
        # if len(latex_question) / len(prompt) < 0.6:
        #     latex_question = prompt
        line['question'] = latex_question
        # line['question_set'] = set(latex_question)
        res_data.append(line)
    return res_data


def get_group_data(data, filter_keyword):
    group_data = defaultdict(list)
    filter_data = [x for x in data if x['keyword'].lower() == filter_keyword]

    res_data = []
    thread_num = 32
    every_thread_data = get_thread_data(filter_data, thread_num)
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        results = [
            executor.submit(get_thread_group_data, thread_data) for thread_data in every_thread_data
        ]
        for future in concurrent.futures.as_completed(results):
            try:
                result = future.result()
                res_data.extend(result)
            except Exception as e:
                print(f"Thread generated an exception: {e}")
    group_data[filter_keyword] = res_data

    for keyword, v in group_data.items():
        print(keyword, len(v))
    return group_data


def get_test_group_data(data, filter_keyword):
    group_data = defaultdict(list)
    for line in tqdm(data, total=len(data), desc='get_test_group_data'):
        keyword = line['keyword'].lower()
        if keyword != filter_keyword:
            continue
        examples = line['example']
        for example in examples:
            question = example['question']
            question = question.replace('%', '\\%')
            latex_question = LatexNodes2Text().latex_to_text(question)
            # if len(latex_question)/len(question) < 0.6:
            #     print('*' * 30)
            #     print(question)
            #     print(latex_question)
            #     latex_question = question

            latex_question = latex_question.replace('\n', '').replace(' ', '')
            example['question'] = latex_question
            example['question_set'] = set(latex_question)
            example['keyword'] = keyword
            group_data[keyword].append(example)
    for keyword, v in group_data.items():
        print(keyword, len(v))

    return group_data


def remove_duplicate(data_path, test_data_list, save_path, keyword):
    if isinstance(data_path, str):
        raw_data = load_json_file(data_path)
    else:
        raw_data = []
        for path in data_path:
            raw_data.extend(path)
    test_data = []
    for test_data_path in test_data_list:
        test_data.extend(load_json_file(test_data_path))

    test_group_data = get_test_group_data(test_data, keyword)
    print(f'test_group_data.keys={test_group_data.keys()}')

    raw_group_data = get_group_data(raw_data, keyword)
    print(f'raw_group_data keys={raw_group_data.keys()}')
    res_data = []
    thread_num = 32

    for keyword, infos in raw_group_data.items():
        # infos = infos[:100]
        now_test_data = test_group_data[keyword]
        every_thread_data = get_thread_data(infos, thread_num, print_log=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
            results = [
                executor.submit(parse_thread_data, thread_data, now_test_data) for thread_data in every_thread_data
            ]
            for future in concurrent.futures.as_completed(results):
                try:
                    result = future.result()
                    res_data.extend(result)
                except Exception as e:
                    print(f"Thread generated an exception: {e}")


    # assert len(res_data) == len(raw_data)

    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)


def filter_with_ratio(data_path, save_path):
    data = load_json_file(data_path)
    common_ratio_distri = []
    for line in data:
        common_ratio_distri.append([line['common_info']['common_ratio_base_custom'], line['common_info']['common_ratio_base_test']])
    common_ratio_distri = pd.DataFrame(common_ratio_distri, columns=['ratio_custom', 'ratio_test'])
    print(common_ratio_distri['ratio_custom'].describe(percentiles=[.9, .8, .7]))
    print(common_ratio_distri['ratio_test'].describe(percentiles=[.9, .8, .7]))

    # for line in data:
    #
    #     common_ratio_distri.append([line['common_info_in_file']['intersection_and_union']])
    # common_ratio_distri = pd.DataFrame(common_ratio_distri, columns=['ratio'])
    # print(common_ratio_distri['ratio'].describe(percentiles=[.9, .8, .7]))

    get_keyword_info(data)
    data = [x for x in data if x['common_info']['common_ratio_base_custom']<0.4 and x['common_info']['common_ratio_base_test']<0.4]

    print(f'after filter data={len(data)}')
    data = filter_in_file_data(data)
    save_json_file(save_path, data)
    get_keyword_info(data)


def filter_in_file_ratio(data_path, save_path):
    data = load_json_file(data_path)
    # save_json_file(save_path, data)
    # return
    common_ratio_distri = []

    for line in data:
        common_ratio_distri.append([line['common_info_in_file']['intersection_and_union']])
    common_ratio_distri = pd.DataFrame(common_ratio_distri, columns=['ratio'])
    print(common_ratio_distri['ratio'].describe(percentiles=[.9, .8, .7]))

    for line in data:
        common_ratio_distri.append([line['common_info_in_file']['common_ratio_base_custom'], line['common_info_in_file']['common_ratio_base_test']])
    common_ratio_distri = pd.DataFrame(common_ratio_distri, columns=['ratio_custom', 'ratio_test'])
    print(common_ratio_distri['ratio_custom'].describe(percentiles=[.9, .8, .7]))
    print(common_ratio_distri['ratio_test'].describe(percentiles=[.9, .8, .7]))

    get_keyword_info(data)
    print(f'文件去重前={len(data)}')

    data = [x for x in data if (x['common_info_in_file']['intersection_and_union'] < 0.9 and x['keyword'] not in ['chinese']) or (
            x['common_info_in_file']['intersection_and_union'] < 0.999 and x['keyword'] in ['chinese']
    )]

    # data = [x for x in data if
    #         x['common_info_in_file']['common_ratio_base_custom'] < 0.8 and x['common_info_in_file']['common_ratio_base_test'] < 0.8]

    print(f'文件内部去重后={len(data)}')
    save_json_file(save_path, data)
    get_keyword_info(data)


if __name__ == '__main__':
    gold_data_path = sys.argv[1]
    test_data_path_list = [
        'questions.json',
    ]
    common_info_save_data_path = sys.argv[2]
    filter_test_save_path = sys.argv[3]
    save_path = sys.argv[4]
    keyword = sys.argv[5]

    remove_duplicate(gold_data_path, test_data_path_list, common_info_save_data_path, keyword)
    filter_with_ratio(common_info_save_data_path, filter_test_save_path)
    filter_in_file_ratio(filter_test_save_path, save_path)
    # check_format(filter_test_save_path)
    pass