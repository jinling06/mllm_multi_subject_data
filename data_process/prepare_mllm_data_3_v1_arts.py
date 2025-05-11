"""
@Time: 2024/9/14 19:18
@Author: xujinlingbj
@File: prepare_mllm_data_3_v1.py
"""
import json
import os
import random
import concurrent.futures
import re
import shutil
import sys
from itertools import chain

from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm

from data_process.image_utils import *
from data_process.mllm_data_3_v1_utils import filter_data, parse_mllm_data_3_thread
from data_process.utils import *

"""
公式图片 带有 formula 关键字
"""


def parse_mllm_data_3_except_chinese(data_dir, json_save_dir):
    if os.path.exists(json_save_dir):
        shutil.rmtree(json_save_dir)
    os.makedirs(json_save_dir, exist_ok=True)
    json_save_dir_name = json_save_dir.split('/')[-1]
    test_data_path = "raw_data/A_test/questions.json"
    group_data = group_raw_test_data(test_data_path)
    print(group_data.keys())
    thread_num = 32
    arts_list = ['history', 'chinese', 'political']
    for subject in os.listdir(data_dir):
        if subject in ['CHINESE'] or 'common_ratio_json_file' in subject:
            continue
        # if subject not in ['POLITICAL_img', 'HISTORY_img']:
        #     continue
        keyword = subject.split('_')[0].lower()
        subject_dir = os.path.join(data_dir, subject)
        # if subject != 'POLITICAL_img':
        #     continue
        if not os.path.isdir(subject_dir):
            continue
        if keyword not in arts_list:
            continue
        print(subject_dir)
        samples = os.listdir(subject_dir)

        every_thread_data = get_thread_data(samples, thread_num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
            for thread_id, thread_data in enumerate(every_thread_data):
                thread_prefix = subject + '_' + str(thread_id)
                now_thread_save_path = os.path.join(json_save_dir, thread_prefix + '.json')
                executor.submit(parse_mllm_data_3_thread, thread_data, group_data, subject_dir, now_thread_save_path,
                                thread_prefix, keyword)


if __name__ == '__main__':
    # 生物、地理、政治、历史
    data_dir = 'MLLM_data_3_v1'
    json_save_dir = f'{data_dir}/common_ratio_json_file_1008_with_formula'
    image_path_2_formular_text_path = f'{data_dir}/image_path_to_formular_text_1008.json'
    change_formular_save_path = f'{data_dir}/train_1027_arts_with_formula_latex.json'
    save_path = f'{data_dir}/train_1110_arts.json'
    only_image_save_path = f'{data_dir}/train_1110_arts_with_image.json'
    # parse_mllm_data_3_except_chinese(data_dir, json_save_dir)

    # change_formular_image_to_text(json_save_dir, image_path_2_formular_text_path,
    #                               change_formular_save_path, source_data_name='mllm_3')
    filter_data(change_formular_save_path, save_path, source_data_name='mllm_3', filter_over_width_image=False)
    filter_with_image(save_path, only_image_save_path)

