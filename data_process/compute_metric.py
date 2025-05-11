"""
@Time: 2024/9/1 16:33
@Author: xujinlingbj
@File: compute_metric.py
"""
import numpy as np

"""
"index": 0D, "index": 1BC, "index": 2D, "index": 3AE, "index": 4B, "index": 5BC, "index": 6B, "index": 7C, "index": 8D, "index": 9A, "index": 10D, "index": 11B, "index": 12D, "index": 13C, "index": 14B, "index": 15D,
"""
from data_process.utils import *


def compute_metric(data_path, label_path):
    test_data = load_json_file(data_path)
    label_data = load_json_file(label_path)
    label_category_to_example = {}
    for line in label_data:
        line['keyword'] = line['keyword']
        label_category_to_example[line['keyword']] = line['example']
    index = 0
    all_score = {}
    keyword_2_multi_ans_num = defaultdict(int)
    for category in test_data:
        keyword = category["keyword"]
        example = category["example"]
        for i in range(len(example)):
            if len(example[i]['model_answer']) > 1:
                keyword_2_multi_ans_num[keyword] += 1
        if keyword not in label_category_to_example:
            continue
        # print(f'开始评估{keyword}分数')

        full_correct_num = 0
        partial_correct_num = 0

        true_example = label_category_to_example[keyword]
        for i in range(len(example)):
            assert true_example[i]['index'] == example[i]['index']
            true_num = len(list(set(true_example[i]['model_answer'])-set(example[i]['model_answer'])))
            if true_num == 0:
                full_correct_num += 1
                continue
            elif true_num < len(true_example[i]['model_answer']):
                partial_correct_num += 1

        #     print(f'pred error index={example[i]["index"]}, '
        #           f'pred_ans = {example[i]["model_answer"]}, true ans={true_example[i]["model_answer"]}')
        # print(f'example num={len(example)}, full_correct_num={full_correct_num},'
        #       f'partial_correct_num={partial_correct_num}')
        now_score = round((full_correct_num+0.5*partial_correct_num)/len(example), 4)
        # print(f'now_score={now_score}')
        all_score[keyword] = now_score
    print(all_score)
    print(f'final score = {np.mean(np.array(list(all_score.values())))}')
    # print(f'每个科目回答多选的分布为: {keyword_2_multi_ans_num}')


if __name__ == '__main__':

    data_path = f'output/submit-llm-427k-stage2-sft-139k-only-image-epoch1_4-arts-plain.json'
    label_path = 'raw_data/A_test/answers_a.json'
    compute_metric(data_path, label_path)
    pass

