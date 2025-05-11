"""
@Time: 2024/9/21 17:42
@Author: xujinlingbj
@File: merge_utils.py
"""
from collections import defaultdict

import numpy as np


def group_data(data):
    res_data = defaultdict(dict)
    for category in data:
        res_data[category['keyword']] = category
    return res_data


def count_differences(lst):
    counts = []
    for i, elem in enumerate(lst):
        count = 0
        for j, other in enumerate(lst):
            if i != j and elem != other:
                count += 1
        counts.append(count)
    return counts


def sort_by_differences(lst):
    differences = count_differences(lst)
    # 使用zip组合原始列表和差异次数，再根据差异次数排序
    combined = list(zip(lst, differences))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=False)
    # 只取排序后的原始元素
    sorted_lst = [x[0] for x in sorted_combined][:12]
    return sorted_lst


def logits_sum_max_ans(ans_list, score_list):

    answer_dict = defaultdict(float)
    for i, (ans, score) in enumerate(zip(ans_list, score_list)):
        # 计算softmax函数
        values = np.array(list(score.values()))
        exp_values = np.exp(values - np.max(values))  # 防止数值溢出
        softmax_values = exp_values / np.sum(exp_values)
        # 将softmax值映射回键值对
        softmax_scores = dict(zip(score.keys(), softmax_values))
        # # 四个选项的概率值都累加，最后求最大
        for key, v in softmax_scores.items():
            # if i == 0:
            #     answer_dict[key] += v * 0.8
            # else:
            answer_dict[key] += v

        # 概率值相加，求最大

        # now_score = softmax_scores.get(ans, 0)
        # answer_dict[ans] += now_score

    max_key = max(answer_dict, key=answer_dict.get)
    return max_key


def compute_entropy(scores):
    # 计算softmax函数
    values = np.array(list(scores.values()))
    values = values/0.01
    exp_values = np.exp(values - np.max(values))  # 防止数值溢出
    softmax_values = exp_values / np.sum(exp_values)

    # 将softmax值映射回键值对
    softmax_scores = dict(zip(scores.keys(), softmax_values))

    # print("软最大值 (Softmax) 结果：")
    # for k, v in softmax_scores.items():
    #     print(f"{k}: {v}")

    # 计算熵函数
    entropy_vals = -softmax_values * np.log(softmax_values)

    # 将熵值映射回键值对
    entropy_scores = dict(zip(scores.keys(), entropy_vals))

    # print("\n熵 (Entropy) 结果：")
    # for k, v in entropy_scores.items():
    #     print(f"{k}: {v}")

    # 求和得到最终结果
    total_entropy = np.sum(entropy_vals)

    # print(total_entropy)
    return total_entropy, softmax_scores


def counter_max_logits_ans(ans_list, score_list):

    raw_data = []

    for ans, score in zip(ans_list, score_list):
        total_entropy, softmax_values = compute_entropy(score)
        raw_data.append({'ans': ans, "score": score, 'entropy': total_entropy,
                         'softmax_values': softmax_values})
    raw_data = sorted(raw_data, key=lambda x: x['entropy'])
    raw_data = raw_data[:1]
    answer_dict = defaultdict(float)
    for x in raw_data:
        score = x['score']
        # 计算softmax函数
        # values = np.array(list(score.values()))
        # exp_values = np.exp(values - np.max(values))  # 防止数值溢出
        # softmax_values = exp_values / np.sum(exp_values)
        # # 将softmax值映射回键值对
        # softmax_scores = dict(zip(score.keys(), softmax_values))
        softmax_scores = x['softmax_values']
        for key, v in softmax_scores.items():
            answer_dict[key] += v
    max_key = max(answer_dict, key=answer_dict.get)
    return max_key