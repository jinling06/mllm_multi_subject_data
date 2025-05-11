"""
@Time: 2024/9/18 18:15
@Author: xujinlingbj
@File: random_test_data.py
"""
import copy
import itertools
import os.path

from data_process.utils import *

raw_test_data_path = 'raw_data/A_test/questions.json'
raw_label_data_path = 'raw_data/A_test/label.json'


def group_data(data):
    res_data = defaultdict(dict)
    for category in data:
        res_data[category['keyword']] = category
    return res_data


for idd in range(24):
    raw_test_data = load_json_file(raw_test_data_path)
    raw_label_data = load_json_file(raw_label_data_path)
    group_test_data = group_data(raw_test_data)
    group_label_data = group_data(raw_label_data)
    res_test_save_path = f'raw_data/A_test/change_data/questions_changed_{idd}.json'
    res_label_save_path = f'raw_data/A_test/change_data/label_changed_{idd}.json'
    base_dir = os.path.dirname(res_test_save_path)
    os.makedirs(base_dir, exist_ok=True)
    res_label_data = []
    res_test_data = []
    str_2_id = {'A':0, 'B': 1, 'C': 2, 'D': 3}
    id_2_str = {v: k for k, v in str_2_id.items()}
    for key, test_info in group_test_data.items():
        now_label_data = group_label_data[key]
        label_example = None
        if now_label_data:
            label_example = now_label_data['example']
        examples = test_info['example']
        new_examples = []
        for i, x in enumerate(examples):
            question = x['question']
            if 'A.' in question and 'E.' not in question and len(question.split('\n')) > 4:
                context = question.split('\n')
                option = context[-4:]
                if 'A.' not in option[0] or 'B.' not in option[1] or 'C.' not in option[2] or 'D.' not in option[3]:
                    new_examples.append(x)
                    continue
                option = [x.replace('A.', '').replace('B.', '').replace('C.', '').replace('D.', '') for x in option]
                origin_option = copy.deepcopy(option)
                ans_choose = []
                if label_example:
                    for label_x in label_example[i]['model_answer']:
                        ans_choose.append(option[str_2_id[label_x]])
                permutations = list(itertools.permutations(option))
                option = list(permutations[idd])
                origin_index = defaultdict(int)
                for j in range(len(option)):

                    origin_index[id_2_str[j]] = id_2_str[origin_option.index(option[j])]
                if len(ans_choose) > 0:
                    changed_ans = []
                    for j in range(len(option)):
                        if option[j] in ans_choose:
                            changed_ans.append(id_2_str[j])
                    assert len(changed_ans) == len(label_example[i]['model_answer'])
                    label_example[i]['model_answer'] = changed_ans
                option[0] = 'A.' + option[0]
                option[1] = 'B.' + option[1]
                option[2] = 'C.' + option[2]
                option[3] = 'D.' + option[3]
                qs = '\n'.join(context[:-4]) +'\n'+ '\n'.join(option)
                x['question'] = qs
                assert len(origin_index) > 0
                x['origin_index'] = origin_index
            new_examples.append(x)
        assert len(new_examples) == len(examples)
        test_info['example'] = new_examples
        res_test_data.append(test_info)
        if now_label_data:
            res_label_data.append(now_label_data)

    save_json_file(res_label_save_path, res_label_data)
    save_json_file(res_test_save_path, res_test_data)

