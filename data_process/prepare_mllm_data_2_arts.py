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
from data_process.prepare_mllm_data_2 import parse_data_thread
from data_process.prepare_mllm_data_v1 import judge_in_test_data
from data_process.prompt_data import chinese_prompt_dic
from data_process.utils import *

"""处理第二个网站爬取的数据"""


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

    files = os.listdir(data_dir)
    science_list = ['history', 'chinese','political']

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
    data_dir = "MLLM_data_2_v2"
    with_common_ratio_save_dir = (
        "MLLM_data_2_v2/train_1027_with_common_ratio_arts"
    )
    save_path = f'{data_dir}/train_1027_arts.json'
    parse_mllm_data_2(data_dir, with_common_ratio_save_dir)
    filter_data(with_common_ratio_save_dir, save_path, source_data_name='mllm_2')
    sample_data(save_path, './output/sample_train_random100.json')
    pass
