"""
@Time: 2024/9/21 17:40
@Author: xujinlingbj
@File: merge_multi_file.py
"""
import os
import sys
from collections import Counter

from analysis_data.merge_utils import group_data, sort_by_differences, counter_max_logits_ans, \
    logits_sum_max_ans
from data_process.compute_metric import compute_metric
from data_process.utils import *


if __name__ == "__main__":
    data_dir = "/Users/xujinlingbj/pycharmWork/game/problem_solving/LLAVA-NEXT-MLLM/output/arts_multi_fold"
    # submit-412k-plain.json
    # raw_pred_path = "/Users/xujinlingbj/pycharmWork/game/problem_solving/LLAVA-NEXT-MLLM/output/merge_24_file_science_412k_0930.json"
    raw_pred_path = "output/merge_24_file_science_412k_0930.json"

    res_path = "output/merge_5_file_arts_134k_0930.json"

    change_res_list = os.listdir(data_dir)

    discard_id_list = []
    change_res_list = [x for x in change_res_list if x.endswith('.json')]

    print(f'候选文件：{len(change_res_list)}')

    res_data = []

    raw_pred_data = load_json_file(raw_pred_path)
    raw_group_data = group_data(raw_pred_data)
    error_score = 0
    for change_pred_path in change_res_list:
        print(change_pred_path)
        change_pred_path = os.path.join(data_dir, change_pred_path)
        # if not change_pred_path.endswith('.json'):
        #     print(f'filter file:{change_pred_path}')
        #     continue
        change_pred_data = load_json_file(change_pred_path)

        change_pred_group_data = group_data(change_pred_data)

        for key, now_pred_data in change_pred_group_data.items():
            for i in range(len(now_pred_data["example"])):
                assert now_pred_data["example"][i]["index"] == raw_group_data[key]["example"][i]["index"]
                ans = now_pred_data["example"][i]["model_answer"][0]

                raw_group_data[key]["example"][i]["model_answer"].append(ans)
                score = now_pred_data["example"][i]["current_score"]

                if 'score' not in raw_group_data[key]["example"][i]:
                    origin_score = raw_group_data[key]["example"][i]['current_score']
                    if origin_score is None:
                        origin_score = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                    raw_group_data[key]["example"][i]['score'] = [origin_score]
                if score is None:
                    print(key, now_pred_data["example"][i])
                    score = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                    error_score += 1

                raw_group_data[key]["example"][i]['score'].append(score)

    res_data = []
    cnt = 0
    for key, value in raw_group_data.items():
        for i in range(len(value["example"])):
            ans_list = value["example"][i]["model_answer"]
            score_list = value["example"][i]["score"]
            # ans_list.pop(0)
            # score_list.pop(0)
            # assert len(score_list) == len(ans_list)
            # assert len(ans_list) == len(change_res_list)

            # ans_list = sort_by_differences(ans_list)
            # frequency = Counter(ans_list)
            # most_common = frequency.most_common(1)
            # most_common = most_common[0][0]

            # most_common = counter_max_logits_ans(ans_list, score_list)
            most_common = logits_sum_max_ans(ans_list, score_list)
            value["example"][i]["model_answer"] = [most_common]
        res_data.append(value)

    save_json_file(res_path, res_data)
    label_path = 'raw_data/A_test/label.json'
    compute_metric(res_path, label_path)
    print(f'error_score={error_score}')
    pass
