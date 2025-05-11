"""
@Time: 2024/8/31 15:49
@Author: xujinlingbj
@File: prepare_mllm_data_2.py
"""
import os
import random
import shutil
import sys

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from data_process.image_utils import *
from data_process.mllm_data_3_v1_utils import filter_data
from data_process.prepare_mllm_data_v1 import judge_in_test_data
from data_process.prompt_data import chinese_prompt_dic
from data_process.utils import *

"""处理第二个网站爬取的数据"""


def parse_data_thread(data, keyword, save_path, thread_prefix):
    res_data = []
    index = 0
    for line in tqdm(data, total=len(data), desc="thread"):
        question = line["question"]
        options = line["options"]
        answer = line["answer"]
        question_img_list = line["question_img_list"]
        option_img_list = line["option_img_list"]
        if len(question) == 0:
            print(f'过滤问题为空的数据：{question}')
            continue
        query_str = question + "".join(options)
        query_str = replace_multiple_spaces(query_str).replace("\n", "")

        if len(query_str) == 0:
            continue
        if check_illegal_answer(answer):
            print(f'非法答案：{answer}')
            continue
        image_list = question_img_list + option_img_list

        if len(image_list) > 1:
            print(f'图片数量大于1：{keyword}')
            continue

        images = []
        save_flag = True
        for img in image_list:
            image_path = os.path.join("llava_sft/math_data", img)
            save_image_path = os.path.join("raw_data/", image_path)
            if not os.path.exists(save_image_path):
                print(f"image not find: {save_image_path}")
                save_flag = False
                break
            images.append(image_path)

        if not save_flag:
            continue
        line["keyword"] = keyword

        line["id"] = f"mllm_data_2_{thread_prefix}_{index}"
        options_str = "\n".join(options)
        image_token = "<image>" * (len(images))
        if len(images) > 0:
            image_token += "\n"

        prompt = f"{image_token}下面是一道【{subject_dic[keyword]}】题，根据问题描述，回答下面的问题。\n问题是：{question}\n选项是:\n{options_str}。\n你的答案是："
        answer = answer.split(",")
        answer = sorted(answer)
        answer = "、".join(answer).strip()
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

        line['conversations'] = conversations
        if len(images) > 0:
            line['image'] = images
        index += 1
        res_data.append(line)

    save_json_file(save_path, res_data)


def parse_file_thread(data_dir, file, keyword, json_save_dir):

    data_path = os.path.join(data_dir, file)
    data = load_json_file(data_path)
    random.shuffle(data)

    num_threads = 32
    every_thred_data = []
    now_thread_data = []
    every_thread_data_num = len(data) // num_threads + 1

    print(f"keyword={keyword}, data num={len(data)}, every_thread_data_num={every_thread_data_num}")
    for line in data:
        now_thread_data.append(line)
        if len(now_thread_data) >= every_thread_data_num:
            every_thred_data.append(now_thread_data)
            now_thread_data = []

    if len(now_thread_data) > 0:
        every_thred_data.append(now_thread_data)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for idx, thred_data in enumerate(every_thred_data):
            thread_prefix = keyword + '_' + str(idx)
            now_thread_save_path = os.path.join(json_save_dir, thread_prefix + '.json')
            executor.submit(parse_data_thread, thred_data, keyword, now_thread_save_path, thread_prefix)


def parse_mllm_data_2(data_dir, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    test_data_path = "raw_data/A_test/questions.json"

    files = os.listdir(data_dir)
    science_list = ['geography', 'math', 'chemistry', 'biology', 'physics']

    for file in files:
        data_path = os.path.join(data_dir, file)
        if not data_path.endswith(".json") or "train" in file:
            continue
        keyword = file.split(".")[0].lower()
        if keyword not in science_list:
            print(f'filter file={file}')
            continue
        parse_file_thread(data_dir, file, keyword, save_dir)


if __name__ == "__main__":
    # 没有公式图
    data_dir = "MLLM_data_2_v2"
    with_common_ratio_save_dir = (
        "MLLM_data_2_v2/train_1027_with_common_ratio"
    )
    image_path_2_formular_text_path = f'{data_dir}/image_path_to_formular_text_1027_science.json'
    change_formular_save_path = f'{data_dir}/train_1027_with_formula_latex.json'
    save_path = f'{data_dir}/train_1027.json'
    # 全部数据，+带公式图的数据
    only_image_save_path = f'{data_dir}/train_1027_with_formula_filter_with_image.json'
    # 只有带公式图的数据
    # formula_only_image_save_path = f'{data_dir}/train_1027_only_formula_filter_with_image.json'
    parse_mllm_data_2(data_dir, with_common_ratio_save_dir)
    change_formular_image_to_text(with_common_ratio_save_dir, image_path_2_formular_text_path,
                                  change_formular_save_path, source_data_name='mllm_2')
    filter_data(change_formular_save_path, save_path, source_data_name='mllm_2')
    filter_with_image(save_path, only_image_save_path)
    # filter_formular_data(only_image_save_path, formula_only_image_save_path)

    sample_data(only_image_save_path, './output/sample_train_random100.json')
    pass
