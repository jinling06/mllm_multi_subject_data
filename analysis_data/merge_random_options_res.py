"""
@Time: 2024/9/19 23:05
@Author: xujinlingbj
@File: merge_random_options_res.py
"""
import base64
import copy
import io
import json
import os
import random
import re
import shutil
import sys
from collections import Counter

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import CLIPImageProcessor

# from data_process.remove_duplicate import filter_same_question
from analysis_data.merge_utils import group_data, sort_by_differences, logits_sum_max_ans
from data_process.compute_metric import compute_metric
from data_process.utils import *


data_dir = "/Users/xujinlingbj/pycharmWork/game/problem_solving/LLAVA-NEXT-MLLM/output/change_data_arts_134k"

change_res_list = os.listdir(data_dir)
change_res_list = sorted(change_res_list, key=lambda x: int(x.split('-')[3]))
change_res_list = [os.path.join(data_dir, x) for x in change_res_list]

raw_test_dir = '/Users/xujinlingbj/pycharmWork/game/problem_solving/LLaVA-MOSS2/raw_data/A_test/change_data'
change_raw_test_list = os.listdir(raw_test_dir)
change_raw_test_list = sorted(change_raw_test_list, key=lambda x: int(x.split('.')[0].split('_')[2]))
change_raw_test_list = [os.path.join(raw_test_dir, x) for x in change_raw_test_list if 'questions' in x]

print(len(change_raw_test_list))
assert len(change_res_list) == len(change_raw_test_list)
raw_pred_path = change_res_list[0]

res_path = "output/merge_24_file_arts_134k_0930.json"

res_data = []
error_score = 0
raw_pred_data = load_json_file(raw_pred_path)
raw_group_data = group_data(raw_pred_data)
for change_test_path, change_pred_path in zip(change_raw_test_list, change_res_list):
    change_test_data = load_json_file(change_test_path)
    change_pred_data = load_json_file(change_pred_path)
    change_test_group_data = group_data(change_test_data)
    change_pred_group_data = group_data(change_pred_data)
    for key, now_pred_data in change_pred_group_data.items():
        now_test_data = change_test_group_data[key]
        for i in range(len(now_pred_data['example'])):
            ans = now_pred_data['example'][i]['model_answer'][0]
            score = now_pred_data["example"][i]["current_score"]
            if score is None:
                print(key, now_pred_data["example"][i])
                score = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                error_score += 1
            if 'origin_index' in now_test_data['example'][i]:
                origin_index = now_test_data['example'][i]['origin_index']
                # if origin_index['A'] != 'A' or origin_index['B'] != 'B' or origin_index['C'] != 'C' or origin_index['D'] != 'D':
                #     print()
                ans = origin_index[ans]
                raw_score = copy.deepcopy(score)
                for k in raw_score.keys():
                    score[k] = raw_score[origin_index[k]]

            raw_group_data[key]['example'][i]['model_answer'].append(ans)
            if 'score' not in raw_group_data[key]["example"][i]:
                origin_score = raw_group_data[key]["example"][i]['current_score']
                if origin_score is None:
                    origin_score = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                raw_group_data[key]["example"][i]['score'] = [origin_score]


            raw_group_data[key]["example"][i]['score'].append(score)

res_data = []
cnt = 0
for key, value in raw_group_data.items():
    for i in range(len(value['example'])):
        ans_list = value['example'][i]['model_answer']
        score_list = value["example"][i]["score"]
        ans_list.pop(0)
        score_list.pop(0)
        assert len(score_list) == len(ans_list)
        assert len(ans_list) == len(change_res_list)

        # ans_list = sort_by_differences(ans_list)
        # frequency = Counter(ans_list)
        # most_common = frequency.most_common(1)
        # most_common = most_common[0][0]

        # most_common = counter_max_logits_ans(ans_list, score_list)
        most_common = logits_sum_max_ans(ans_list, score_list)
        value['example'][i]['model_answer'] = [most_common]
    res_data.append(value)

save_json_file(res_path, res_data)
label_path = 'raw_data/A_test/label.json'
compute_metric(res_path, label_path)
print(cnt)
