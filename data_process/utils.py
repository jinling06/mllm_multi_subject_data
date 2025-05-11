"""
@Time: 2024/8/26 12:55
@Author: xujinlingbj
@File: utils.py
"""
import copy
import difflib
import json
import os.path
import random
import re
import sys
from collections import defaultdict
from typing import List
import requests
from urllib.parse import urlparse, urljoin
from os.path import basename
import numpy as np
import pandas as pd
from PIL import Image
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

subject_dic = {
    'none': '未知',
    'geography': '地理',
    'physics': '物理',
    'math': '数学',
    'chemistry': '化学',
    'history': '历史',
    'biology': '生物',
    'political': '政治',
    'chinese': '语文',
               }


common_ratio_dic = {
'geography': 0.65,
    'physics': 0.5,
    'math': 0.8,
    'chemistry': 0.7,
    'history': 0.8,
    'biology': 0.8,
    'political': 0.8,
    'chinese': 0.8,
}

def count_multi_questions(input_string):
    # 定义要匹配的选项字符（注意这里用转义序列来匹配 `.` 符号）
    options = ['A\\.', 'B\\.', 'C\\.', 'D\\.']

    # 存储每个选项的计数
    count_dict = {option[:-2]: 0 for option in options}  # 去掉`\.`部分

    for option in options:
        # 统计每个选项的次数
        matches = re.findall(option, input_string)
        count_dict[option[:-2]] = len(matches)

    for value in count_dict.values():
        if value > 2:
            return True
    return False


def download_image(image_url, save_path):
    # 解析URL以获取图像文件名
    # parsed_url = urlparse(image_url)

    try:
        # 发送HTTP GET请求以下载图像
        response = requests.get(image_url)
        response.raise_for_status()  # 如果产生HTTP错误，则引发异常
        # 将图像写入本地文件
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"下载图片时发生错误: {e}")
        return False


def sample_test_b_data(data_path, output_path):
    data = load_json_file(data_path)
    print(f'{data_path}, data num={len(data)}')
    keyword2data_list = defaultdict(list)

    for x in data:
        if 'image' not in x:
            continue
        keyword = x['keyword']
        if len(keyword2data_list[keyword]) < 8000:
            x['index'] = x['id']
            x['prompt'] = x['conversations'][0]['value']
            x['origin_index'] = x['id']
            keyword2data_list[keyword].append(x)
    res_data = []
    for key, v in keyword2data_list.items():
        print(key, len(v))
        res_data.extend(v)

    save_json_file(output_path, res_data)
    print(f'save path=  {output_path}')


def remove_specific_subject_data(data_path, subject_name: List):
    data = load_json_file(data_path)

    science_list = ['geography', 'math', 'chemistry', 'biology', 'physics', 'history', 'chinese','political']

    science_data = []
    for line in tqdm(data, total=len(data), desc='split_art_science'):
        if line['keyword'] in subject_name:
            continue
        science_data.append(line)

    data_path = data_path.split('/')
    data_dir = '/'.join(data_path[:-1])
    file_name = data_path[-1].split('.')[0]
    subject_str = '_'.join(subject_name)
    science_save_path = os.path.join(data_dir, file_name+ f'_rm_{subject_str}.json')
    save_json_file(science_save_path, science_data)


def split_4_fold_data(data_path):
    data = load_json_file(data_path)
    file_dir = os.path.dirname(data_path)
    file_name = os.path.basename(data_path)
    print(f'file_dir={file_dir}')
    print(f'file_name={file_name}')
    print(f'data num={len(data)}')
    random.shuffle(data)
    num = len(data) // 4 + 1
    for i in range(4):
        start = i*num
        end = i*num+num
        print(start, end)
        now_data = data[: start] + data[end:]
        save_path = os.path.join(file_dir, f'fold_{i}_{file_name}')
        save_json_file(save_path, now_data)


def jaccard_similarity(set_a, set_b):

    intersection = sum(1 for char in set_a if char in set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union


def judge_in_test_data(test_data, text, event, min_ratio=0.5):
    max_common_info = {
        "common_text": "",
        "test_query": "",
        "mllm_text": "",
        "common_ratio_base_custom": 0.0,
        "common_ratio_base_test": 0.0,
    }
    for line in test_data:
        query = line["question"]
        query = query.replace("\n", "")
        common_text = LCS_with_difflib(query, text)
        common_ratio_base_test = len(common_text) / (len(query) + 1e-6)
        common_ratio_base_custom = len(common_text) / (len(text) + 1e-6)
        max_ratio = max(common_ratio_base_custom, common_ratio_base_test)

        # if common_ratio_base_custom > max_common_info['common_ratio_base_custom'] or common_ratio_base_test > max_common_info['common_ratio_base_test']:
        if max_ratio > max(max_common_info["common_ratio_base_custom"], max_common_info["common_ratio_base_test"]):
            max_common_info["common_text"] = common_text
            max_common_info["test_query"] = query
            max_common_info["mllm_text"] = text
            max_common_info["common_ratio_base_custom"] = common_ratio_base_custom
            max_common_info["common_ratio_base_test"] = common_ratio_base_test
        if max_common_info["common_ratio_base_test"] > min_ratio or max_common_info["common_ratio_base_custom"] > min_ratio:
            event.set()

            return max_common_info

    return max_common_info


# 移动类似 remove_duplicate_with_intersection_and_union 的计算到单独的函数中，以便于线程处理
def duplicate_in_file_worker_function(test_data, text, text_set, event, min_ratio=0.5):
    max_common_info = {
        'common_text': '',
        "intersection_and_union": 0.0,
        "lcs_common_text": "",
        "lcs_test_query": "",
        "lcs_mllm_text": "",
        "common_ratio_base_custom": 0.0,
        "common_ratio_base_test": 0.0,
    }
    # 遍历每一行数据
    for line in test_data:
        if event.is_set():
            return max_common_info
        query = line['question']
        question_set = line['question_set']

        # 计算相似度
        ratio = jaccard_similarity(question_set, text_set)

        if ratio > max_common_info['intersection_and_union']:
            max_common_info['intersection_and_union'] = ratio
            max_common_info['common_text'] = line['question']
        if ratio < 0.5:
            continue
        common_text = LCS_with_difflib(query, text)
        common_ratio_base_test = len(common_text) / (len(query) + 1e-6)
        common_ratio_base_custom = len(common_text) / (len(text) + 1e-6)
        max_ratio = max(common_ratio_base_custom, common_ratio_base_test)

        if max_ratio > max(max_common_info["common_ratio_base_custom"], max_common_info["common_ratio_base_test"]):
            max_common_info["lcs_common_text"] = common_text
            max_common_info["lcs_test_query"] = query
            max_common_info["lcs_mllm_text"] = text
            max_common_info["common_ratio_base_custom"] = common_ratio_base_custom
            max_common_info["common_ratio_base_test"] = common_ratio_base_test
        if max_common_info["common_ratio_base_test"] > min_ratio or max_common_info["common_ratio_base_custom"] > min_ratio:
            event.set()
            return max_common_info

    return max_common_info


def remove_duplicate_with_intersection_and_union_in_test(test_data, text, min_ratio=0.8):
    # max_common_info = {
    #     'common_text': '',
    #     "intersection_and_union": 0.0,
    # }
    max_common_info = {
        "common_text": "",
        "test_query": "",
        "mllm_text": "",
        "common_ratio_base_custom": 0.0,
        "common_ratio_base_test": 0.0,
    }
    num_threads = 32
    every_thread_data = get_thread_data(test_data, num_threads=num_threads)
    # 使用线程事件来管理线程之间的状态共享
    event = threading.Event()

    # 使用 ThreadPoolExecutor 来管理线程池
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(judge_in_test_data, thread_data, text, event, min_ratio) for thread_data in
                   every_thread_data]
        # 确保每个线程运行完成
        for future in as_completed(futures):
            # if event.is_set():
            #     break
            result = future.result()
            # if result['intersection_and_union'] > max_common_info['intersection_and_union']:
            #     max_common_info = result
            max_ratio = max(result['common_ratio_base_custom'], result['common_ratio_base_test'])
            if max_ratio > max(max_common_info['common_ratio_base_custom'], max_common_info['common_ratio_base_test']):
                max_common_info = result

    return max_common_info


def remove_duplicate_with_intersection_and_union_in_file(test_data, text, text_set, min_ratio=0.8):
    max_common_info = {
        'common_text': '',
        "intersection_and_union": 0.0,
        "lcs_common_text": "",
        "lcs_test_query": "",
        "lcs_mllm_text": "",
        "common_ratio_base_custom": 0.0,
        "common_ratio_base_test": 0.0,
    }

    num_threads = 32
    every_thread_data = get_thread_data(test_data, num_threads=num_threads)
    # 使用线程事件来管理线程之间的状态共享
    event = threading.Event()

    # 使用 ThreadPoolExecutor 来管理线程池
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(duplicate_in_file_worker_function, thread_data, text, text_set, event, min_ratio) for thread_data in
                   every_thread_data]
        # 确保每个线程运行完成
        for future in as_completed(futures):

            result = future.result()
            # if result['intersection_and_union'] > max_common_info['intersection_and_union']:
            #     max_common_info = result
            max_ratio = max(result['common_ratio_base_custom'], result['common_ratio_base_test'])
            if max_ratio > max(max_common_info['common_ratio_base_custom'], max_common_info['common_ratio_base_test']):
                max_common_info = result

    return max_common_info


def find_positions(string, substring):
    """
    找到字符串中所有子字符串出现的位置。
    """
    positions = []
    pos = string.find(substring)
    while pos != -1:
        positions.append(pos)
        pos = string.find(substring, pos + 1)
    return positions


def replace_at_positions(string, positions, replacements):
    """
    在指定的位置将字符串替换为给定的替换字符串。
    """
    if len(positions) != len(replacements):
        raise ValueError("positions 和 replacements 的长度必须一致")

    # 为了方便替换，逆序替换避免位置偏移
    for i in range(len(positions) - 1, -1, -1):
        pos = positions[i]
        replacement = replacements[i]
        string = string[:pos] + replacement + string[pos + len("<image>"):]

    return string


def check_illegal_formular_text(model_pred):
    if 'overrightarrow' in model_pred or '\\vec' in model_pred or 'overline' in model_pred:
        return True
    if 'cot' in model_pred or 'backprime' in model_pred:
        return True
    return False


def change_formular_image_to_text(raw_data_dir, image_to_formular_text_path, save_path, source_data_name='mllm_3'):
    raw_data = []
    for raw_data_path in os.listdir(raw_data_dir):
        raw_data_path = os.path.join(raw_data_dir, raw_data_path)
        print(raw_data_path)
        raw_data.extend(load_json_file(raw_data_path))
    image_to_formular_text_data = load_json_file(image_to_formular_text_path)
    image_path_to_formular_text = defaultdict(dict)
    for line in image_to_formular_text_data:
        # model_pred clean_text
        if check_illegal_formular_text(line['model_pred']):
            print(f'过滤预测错误的公式图')
            continue
        image_path_to_formular_text[line['image_path']] = line

    print(f'data num={len(raw_data)}, image_path_to_formular_text num={len(image_path_to_formular_text)}')
    res_data = []
    for line in tqdm(raw_data, total=len(raw_data), desc='replace_latex'):
        save_flag = True
        formular_data = False
        if 'image' in line:
            if isinstance(line['image'], str):
                image_path = line['image']
                image_path = os.path.join('raw_data', image_path)
                # image = Image.open(image_path)
                if judge_formula_image(image_path, source=source_data_name):
                    formular_info = image_path_to_formular_text[image_path]

                    if len(formular_info) == 0:
                        save_flag = False
                        continue
                    formular_text = formular_info['clean_text']
                    if len(formular_text) == '':
                        save_flag = False
                        continue
                    line.pop('image')
                    prompt = line['conversations'][0]['value']
                    line['origin_prompt'] = copy.deepcopy(prompt)
                    prompt = prompt.replace('<image>', formular_text)
                    line['formular_info'] = formular_info

                    assert prompt.count('<image>') == 0
                    assert 'image' not in line
                    formular_data = True
                    line['conversations'][0]['value'] = prompt
            elif isinstance(line['image'], list):
                new_image_list = []
                prompt = line['conversations'][0]['value']
                formular_image_infos = []
                formular_image_index_list = []
                replacements = []
                for i, image_path in enumerate(line['image']):
                    save_image_path = os.path.join('raw_data', image_path)
                    # image = Image.open(save_image_path)
                    if judge_formula_image(save_image_path, source=source_data_name):
                        formular_info = image_path_to_formular_text[save_image_path]

                        if len(formular_info) == 0:
                            save_flag = False
                            break

                        formular_text = formular_info['clean_text']
                        if len(formular_text) == '':
                            save_flag = False
                            break
                        formular_image_index_list.append(i+1)
                        replacements.append(formular_text)
                        formular_data = True
                        formular_image_infos.append(formular_info)
                        continue
                    new_image_list.append(image_path)
                if not save_flag:
                    continue
                # 更新prompt
                if len(replacements) > 0:

                    # 找到所`有 <image> 的位置
                    image_positions = find_positions(prompt, "<image>")

                    # 提取需要替换的 <image> 的实际位置
                    actual_positions = [image_positions[i - 1] for i in
                                        formular_image_index_list]  # positions_to_replace is 1-based

                    # 替换指定位置的 <image>
                    line['origin_prompt'] = copy.deepcopy(prompt)
                    prompt = replace_at_positions(prompt, actual_positions, replacements)
                    assert prompt.count('<image>') == len(new_image_list)

                    line['conversations'][0]['value'] = prompt
                    line['formular_info'] = formular_image_infos

                if len(new_image_list) == 0:
                    line.pop('image')
                    assert 'image' not in line
                else:
                    line['image'] = new_image_list

        if save_flag:
            line['formular_data'] = formular_data
            res_data.append(line)
    save_json_file(save_path, res_data)


def split_arts_science_data(data_path):
    data = load_json_file(data_path)
    arts_list = ['history', 'chinese','political']
    # 'physics',
    science_list = ['geography', 'math', 'chemistry', 'biology', 'physics']
    arts_data = []
    science_data = []
    for line in tqdm(data, total=len(data), desc='split_art_science'):
        if line['keyword'] in science_list:
            science_data.append(line)
        else:
            arts_data.append(line)
    data_path = data_path.split('/')
    data_dir = '/'.join(data_path[:-1])
    file_name = data_path[-1].split('.')[0]
    arts_save_path = os.path.join(data_dir, file_name+'_arts.json')
    science_save_path = os.path.join(data_dir, file_name+ '_science.json')
    save_json_file(arts_save_path, arts_data)
    save_json_file(science_save_path, science_data)


def split_specific_subject_data(data_path, subject_name: List):
    data = load_json_file(data_path)

    science_list = ['geography', 'math', 'chemistry', 'biology', 'physics', 'history', 'chinese','political']
    arts_data = []
    science_data = []
    for line in tqdm(data, total=len(data), desc='split_art_science'):
        if line['keyword'] in subject_name:
            science_data.append(line)

    image_data = []
    without_image_data = []
    for line in science_data:
        if 'image' in line:
            image_data.append(line)
        else:
            without_image_data.append(line)
    random.shuffle(image_data)
    random.shuffle(without_image_data)
    print(f'image_data num={len(image_data)}, without_image_data num={len(without_image_data)} ')
    data_num = 40000
    res_data = image_data[:data_num]
    res_data.extend(without_image_data[:data_num-len(image_data)])
    data_path = data_path.split('/')
    data_dir = '/'.join(data_path[:-1])
    file_name = data_path[-1].split('.')[0]
    subject_str = '_'.join(subject_name)
    science_save_path = os.path.join(data_dir, file_name+ f'_{subject_str}_40k.json')
    save_json_file(science_save_path, res_data)


def judge_formula_image(image_path=None, open_image=None, source='mllm_3'):
    assert open_image or image_path

    if open_image:
        image = open_image
    else:
        try:
            image = Image.open(image_path)
        except:
            # 返回True ，直接将这条数据过滤掉
            print(f'open error: {image_path}')
            return True
    if source == 'mllm_3' and 'formula' in image_path:
        return True
    if source in ['mllm_1', 'mllm_2']:
        if image.height < 50:
            return True

    return False


def filter_with_image(data_path, save_path):
    image_prefix = 'raw_data'
    res_data = []
    data = load_json_file(data_path)
    print(f'raw data num={len(data)}')
    cnt = 0
    for line in tqdm(data, total=len(data), desc='filter_w_image'):
        save_flag = True
        if 'image' in line:
            res_data.append(line)

    print(f'res data num={len(res_data)}')
    save_json_file(save_path, res_data)
    get_keyword_info(res_data)


def filter_formular_data(data_path, save_path):
    image_prefix = 'raw_data'
    res_data = []
    data = load_json_file(data_path)
    print(f'raw data num={len(data)}')
    cnt = 0
    for line in tqdm(data, total=len(data), desc='filter_w_image'):

        if 'formular_data' in line and line['formular_data']:
            res_data.append(line)

    print(f'res data num={len(res_data)}')
    save_json_file(save_path, res_data)
    get_keyword_info(res_data)


def del_formular_data(data_path, save_path):

    res_data = []
    data = load_json_file(data_path)
    print(f'raw data num={len(data)}')
    cnt = 0
    for line in tqdm(data, total=len(data), desc='del_formular_data'):

        if 'formular_data' in line and line['formular_data']:
            continue
        res_data.append(line)

    print(f'res data num={len(res_data)}')
    save_json_file(save_path, res_data)
    get_keyword_info(res_data)

def filter_without_image(data_path):
    image_prefix = 'raw_data'
    res_data = []
    data = load_json_file(data_path)
    print(f'raw data num={len(data)}')
    cnt = 0
    for line in tqdm(data, total=len(data), desc='filter_w_image'):
        save_flag = True
        if 'image' not in line:
            res_data.append(line)
    print(f'res data num={len(res_data)}')
    file_dir = '/'.join(data_path.split('/')[:-1])
    file_name = data_path.split('/')[-1].split('.')[0]
    save_path = os.path.join(file_dir, file_name + '_without_image.json')
    save_json_file(save_path, res_data)

def get_text_box_by_coordinates(coordinates: List[List[int]]):
    x_list = [x for x, y in coordinates]
    y_list = [y for x, y in coordinates]
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def parse_ocr_text(ocr_text_res):
    ocr_results = []

    for line in ocr_text_res:
        coordinates = line[0]
        text = line[1][0]
        proba = line[1][1]
        ocr_results.append(
            {
                "bbox": get_text_box_by_coordinates(coordinates),
                "text": text,
                "proba": proba
            }
        )

    return ocr_results



def remove_leading_numbering(text):
    # 匹配前导序号，例如 "3. " 或 "3．" 或 "3)"
    pattern = r'^\d+\s*[．.。)]\s*'
    return re.sub(pattern, '', text)



def base64_to_image(base64_str):
    """
    将Base64编码的图像转换为PIL图像对象

    :param base64_str: 字符串形式的Base64编码的图像
    :return: PIL图像对象
    """
    # 解码Base64字符串
    image_data = base64.b64decode(base64_str)

    image = Image.open(io.BytesIO(image_data))

    return image


def compute_answer_num(text):
    # 使用 re.findall 查找所有的 "ABCD" 子字符串
    matches = re.findall(r'[ABCD]', text)

    # 计算匹配的个数
    num_matches = len(matches)
    return num_matches


def get_keyword2id():
    keyword2id = {}
    for k, v in subject_dic.items():
        keyword2id[k] = len(keyword2id)
    return keyword2id

keyword2id = get_keyword2id()

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    match = pattern.search(text)
    return match is not None


def get_thread_data(data, num_threads=32, print_log=False):
    every_thred_data = []
    now_thread_data = []
    every_thread_data_num = len(data) // num_threads + 1
    if print_log:
        print(f"data num={len(data)}, every_thread_data_num={every_thread_data_num}")

    for line in data:
        now_thread_data.append(line)
        if len(now_thread_data) >= every_thread_data_num:
            every_thred_data.append(now_thread_data)
            now_thread_data = []

    if len(now_thread_data) > 0:
        every_thred_data.append(now_thread_data)
    return every_thred_data


def get_thread_data_with_image_num(data, num_threads=32, print_log=False):
    every_thread_data = [[] for _ in range(num_threads)]
    image_data = []
    wo_image_data = []

    # 分类数据
    for line in data:
        if 'image' in line:
            image_data.append(line)
        else:
            wo_image_data.append(line)

    # 计算每个线程需要分配的数据量
    thread_image_num = len(image_data) // num_threads + 1
    thread_wo_image_num = len(wo_image_data) // num_threads + 1

    if print_log:
        print(f"data num={len(data)}, thread_image_num={thread_image_num}, thread_wo_image_num={thread_wo_image_num}")
    # 平均分配 wo_image_data 到每个线程
    for i in range(num_threads):
        start_index = i * thread_wo_image_num
        end_index = start_index + thread_wo_image_num
        every_thread_data[i].extend(wo_image_data[start_index:end_index])

    # 平均分配 image_data 到每个线程
    for i in range(num_threads):
        start_index = i * thread_image_num
        end_index = start_index + thread_image_num
        every_thread_data[i].extend(image_data[start_index:end_index])


    return every_thread_data


def split_multi_question_text(text):

    # 使用正则表达式找到所有小题，保留编号和内容
    if re.search(r'【\d+】', text):
        outline_match = re.search(r'【题目】(.*?)(?=【\d+】|(\(1\)))', text, re.S)
        outline = outline_match.group(1).strip() if outline_match else ""

        questions = re.findall(r'【(\d+)】([^【]*)', text)
    elif re.search(r'（\d+）', text):
        outline_match = re.search(r'【题目】(.*?)(?=（\d+）|(\(1\)))', text, re.S)
        outline = outline_match.group(1).strip() if outline_match else ""
        questions = re.findall(r'\（(\d+)\）(.*?)(?=\（\d+\）|$)', text, re.S)
    else:
        outline_match = re.search(r'【题目】(.*?)(?=(\d+)|(\(1\)))', text, re.S)
        outline = outline_match.group(1).strip() if outline_match else ""
        questions = re.findall(r'\((\d+)\)(.*?)(?=\(\d+\)|$)', text, re.S)

    # 输出题目大纲和每个子问题
    multi_question_list = []
    for question in questions:
        question_number, question_content = question
        format_question = f"{outline} {question_content.strip()}"
        format_question = format_question.strip()
        multi_question_list.append(format_question)
    return multi_question_list


def has_other_than_uppercase(s):
    #
    return any(not char.isupper() and not char.isspace() for char in s)


def format_answer(s):
    # if contains_other_uppercase(s) or contains_chinese(s):
    #     print(f'包含自他非法字母：{s}')
    #     return ''
    if contains_uppercase_abcd(s):
        s = s.replace('I', '')
        ans = [char for char in s if char.isupper()]
        ans = list(set(ans))
        ans = sorted(ans)
        if len(ans) > 4:
            return ''
        s = '、'.join(ans)
        return s

    return ''


def replace_multiple_spaces(text):
    # 使用正则表达式将多个空格替换成一个空格
    return re.sub(r'\s+', ' ', text)


def LCS_with_difflib(s1, s2):
    # 创建 SequenceMatcher 对象
    seq_matcher = difflib.SequenceMatcher(None, s1, s2)

    # 获取编辑操作
    opcodes = seq_matcher.get_opcodes()

    # 收集最长公共子序列中的字符
    lcs = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            lcs.append(s1[i1:i2])

    return ''.join(lcs)


def get_first_abcd_index(text):
    legal_ans_list = ["A", "B", "C", "D"]
    index_list = []
    for legal_ans in legal_ans_list:
        index = text.find(legal_ans)
        if index != -1:
            index_list.append([legal_ans, index])
    index_list = sorted(index_list, key=lambda x: x[1])
    return index_list

def is_all_uppercase(s):
    if not s:  # 判断字符串是否为空
        return False
    return s.isupper() and s.isalpha()


def parse_answer_from_custom_data(text):
    pattern_list = [r'【答案】(.*?)【解析】', r'试题答案练习册答案在线课程(.*?)分析',
                    r'试题答案练习册答案在线课程(.*?)【解析】',
                    r'答案:(.*?)', r'答案：(.*?)',
                    r'题答案练习册答案在线课程(.*?)解析', r'故选(.*?)']
    answer = ""
    for i, pattern_text in enumerate(pattern_list):
        pattern = re.compile(pattern_text)
        match = pattern.search(text)
        if match:

            answer = match.group(1).strip().replace(' ', '').replace('.', '')
            if i > 0 and not is_all_uppercase(answer):
                # print(answer, ' ++++  ', text)
                # print(len(answer))
                answer = ""
                continue
            else:
                break
    if answer == '':
        text = text.replace('试题答案练习册答案在线课程', '').replace(' ', '').replace('.', '')
        if len(text) == 1 and is_all_uppercase(text):
            answer = text
    return answer


def replace_last_str(text, option_str, replace_text):
    last_index = text.rfind(option_str)
    if last_index != -1:
        text = text[:last_index] + replace_text + text[last_index + len(option_str):]
    return text


def insert_abcd_in_item(text):
    # # 正则表达式匹配 A, B, C, D 前面的位置，并添加换行符
    # pattern = re.compile(r'(?=[ABCDEFGH][\.\．\、])')
    # # pattern = re.compile(r'(?=[ABCDEFGH]\．)')
    # # 替换匹配到的位置前面加上换行符
    # modified_text = pattern.sub('\n', text)
    option_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    punctuation_list = ['.', '．', '、']
    for option in option_list:
        for puc in punctuation_list:
            match_str = option+puc
            text = replace_last_str(text, match_str, "\n"+match_str)
    return text


def get_mavis_choose_prompt(question, choose, image_num):
    image_token = '<image>'*image_num
    text = f'{image_token}\nAccording to the question shown in the image, please first perform reasoning, then finally select the right answer from the choices, e.g., Answer: xxx.\nQuestion: {question}\nChoices:{choose}'
    return text


def get_mavis_essay_prompt(question, image_num):
    image_token = '<image>' * image_num
    text = f'{image_token}\nAccording to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., The answer is xxx\nQuestion: {question}'
    return text

import bisect


def find_first_greater_than(sorted_arr, target):
    # 确保列表是排序的
    # sorted_arr = sorted(arr)

    # 使用 bisect_right 找到第一个大于 target 的位置
    index = bisect.bisect_right(sorted_arr, target)

    # 检查是不是超出了列表长度
    if index == len(sorted_arr):
        return -1  # 表示找不到

    return index


def check_illegal_answer(text):

    if len(text) > 10 or len(text) == 0:
        return True
    return False


def load_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    # print(f'{path}={len(data)}')
    return data


def load_jsonl_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = json.loads(data[i])
    return data


def save_jsonl_data(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')


def contains_other_uppercase(s):
    # 定义正则表达式，匹配除了ABCDE之外的其他大写字母
    pattern = re.compile(r'[F-Z]')
    # 使用re.search()查找是否有匹配的字符
    if pattern.search(s):
        return True
    return False


def contains_uppercase_abcd(input_string):
    # 定义包含大写的A、B、C或D的正则表达式模式
    pattern = r'[ABCD]'

    # 使用re.search()函数查找模式是否存在于输入字符串中
    match = re.search(pattern, input_string)

    # 如果找到匹配，返回True；否则，返回False
    return bool(match)

def contain_all_abcd(text):
    if 'A' not in text:
        return False
    if 'B' not in text:
        return False
    if 'C' not in text:
        return False
    if 'D' not in text:
        return False
    return True


def save_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'{path}= {len(data)}')


def compute_two_file_diff(file_list):
    data_list = []
    for file_path in file_list:
        data = load_json_file(file_path)
        keyword2example = defaultdict(list)
        for line in data:
            if line['keyword'] not in ['Geography', 'Physics']:
                keyword2example[line['keyword']] = line['example']
        data_list.append(keyword2example)
    diff_num = 0
    for k, examples in data_list[0].items():
        for i in range(len(examples)):
            example = examples[i]
            flag = False
            for j in range(1, len(data_list)):
                try:
                    now_example = data_list[j][k][i]
                except:
                    print(len(examples), len(data_list[j][k]))
                    print(data_list[j][k])
                    print(examples)
                if example['model_answer'] !=now_example['model_answer']:
                    if example['index'] != now_example['index']:
                        print(example)
                        print(data_list[j])
                        continue
                    flag = True
                    break
            if flag:
                diff_num += 1
                print(k, example)
    print(f'diff_num={diff_num}')


def compute_json_file_line_num(data_list):
    for data_path in data_list:
        data = load_json_file(data_path)
        print(f'{data_path}, num={len(data)}')


def sample_data(data_path):
    data = load_json_file(data_path)
    print(f'{data_path}, data num={len(data)}')
    # data = [x for x in data if 0.3<x['common_info']['common_ratio']<=0.4]
    # data = [x for x in data if 'business' in x['id']]
    category = defaultdict(int)
    # data_dict = defaultdict(int)
    cnt = 0
    random.shuffle(data)
    res_data = []
    length_distri = []
    no_image_category = defaultdict(int)
    for x in data:
        if 'image' not in x:
            cnt += 1
            no_image_category[x['keyword']] += 1
        category[x['keyword']] += 1
        conversations = x['conversations']
        query = conversations[0]['value']
        answer = conversations[1]['value']
        length_distri.append([len(query) + len(answer), len(query) + len(answer)+ 576])
        if category[x['keyword']] < 100:
            res_data.append(x)

    print(category)
    print(f'不带图的数量={cnt}')
    print('不带图的分布：')
    print(no_image_category)
    print(f'数据长度分布')
    columns = ['len', 'add_image_len']
    length_distri = pd.DataFrame(length_distri, columns=columns)
    for col in columns:
        print(length_distri[col].describe(percentiles=[.99, .95, .9, .8, .7]))


def reformat_llava_558k(data_path, save_path):
    data = load_json_file(data_path)
    print(f'data num={len(data)}')
    for line in data:
        line['image'] = os.path.join('LLaVA-Pretrain/images', line['image'])
    save_json_file(save_path, data)


def merge_multi_file(file_list, save_path):
    res_data = []
    for file_path in file_list:
        res_data.extend(load_json_file(file_path))
        print(f'{file_path}, num={len(res_data)}')
    res_data = [x for x in res_data if 'image' not in x]
    random.shuffle(res_data)
    category = defaultdict(int)
    for x in res_data:
        category[x['keyword']] += 1
    print(category)

    save_json_file(save_path, res_data)
    print(f'save file={save_path}, num={len(res_data)}')


def get_keyword_info(data):
    category = defaultdict(int)

    no_image_infos = defaultdict(int)
    for x in data:
        if 'image' not in x:
            no_image_infos[x['keyword']] += 1
        category[x['keyword']] += 1
    print(f'no image distribution={no_image_infos}')
    print(category)


def get_length_distribute(data):
    length_dis = []
    for line in data:
        prompt = line['conversations'][0]['value']
        answer = line['conversations'][1]['value']
        length_dis.append([len(prompt+answer)])
    length_dis = pd.DataFrame(length_dis, columns=['len'])
    print(length_dis['len'].describe(percentiles=[.99, .95, .9, .8, .7]))


def judge_illegal_image(open_image: Image):
    if open_image.width < 50 or open_image.height < 50:
        return True
    if open_image.width > 1500 or open_image.height > 1000:
        return True
    return False


def statistics_image_width_height_distribution(data, image_dir):
    len_distri = []
    for line in data:
        if 'image' in line:
            image_list = line['image']
            if isinstance(image_list, str):
                image_list = [image_list]
            for image_path in image_list:
                image_path = os.path.join(image_dir, image_path)
                image = Image.open(image_path)
                len_distri.append([image.width, image.height])
    columns = ['width', 'height']
    len_distri = pd.DataFrame(len_distri, columns=columns)
    for x in columns:
        print(len_distri[x].describe(percentiles=[0.99, 0.95, 0.9, 0.8, 0.7]))


def get_next_char(char):
    """
    Given a character, return the next character in the alphabet.
    If the input is 'A', the output is 'B', and so on.
    """
    if char.isalpha() and len(char) == 1:
        # Convert the character to its ASCII value
        char_value = ord(char)
        # Increment the ASCII value by 1 to get the next character
        next_char_value = char_value + 1
        # Convert the ASCII value back to a character
        next_char = chr(next_char_value)
        return next_char
    else:
        raise ValueError("Input must be a single alphabetic character")