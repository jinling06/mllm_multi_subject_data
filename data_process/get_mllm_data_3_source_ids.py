"""
@Time: 2024/10/19 21:36
@Author: xujinlingbj
@File: get_mllm_data_3_source_ids.py
"""
import os

from tqdm import tqdm
from data_process.utils import *


def iterate_mllm_data_3_thread(samples, version_dir, keyword):
    error_num = 0
    all_num = 0
    index = 0

    res_data = []
    for sample_name in tqdm(samples, total=len(samples), desc="sample"):
        sample_dir = os.path.join(version_dir, sample_name)
        if not os.path.isdir(sample_dir):
            print(f'error sample_dir={sample_dir}')
            continue
        source_data_id = ''
        image_files = []
        for file_name in os.listdir(sample_dir):
            file_path = os.path.join(sample_dir, file_name)
            if "answer" in file_name:
                source_data_id = file_name.split('.')[0].split('_')[3]
            if file_path.endswith(".html"):
                html_file_path = file_path
            elif file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                if "answer" not in file_name:
                    image_files.append(file_path)
            elif file_path.endswith(".svg"):
                if "answer" not in file_name:
                    png_file_path = os.path.join(sample_dir, file_name.split(".")[0] + ".png")
                    try:
                        # if not os.path.exists(png_file_path):
                        #     image = open_svg(file_path)
                        #     image.save(png_file_path)
                        image_files.append(png_file_path)
                    except:
                        print(f"paser svg error:{file_path}")

            elif file_path.endswith(".json"):
                if "answer_0_Answer" in file_path:
                    answer_file_path = file_path
                elif "answer_1_Parse" in file_path:
                    parse_file_path = file_path
            else:
                raise ValueError(f"文件格式错误：{file_path}")
        image_files = list(set(image_files))

        res_data.append([source_data_id, keyword, len(image_files)])
    return res_data


if __name__ == '__main__':
    data_dir_list = [
        'MLLM_data_3_v1',
        'MLLM_data_3_v1/CHINESE',
        'MLLM_data_3_v2',
        'MLLM_data_3_v3',
        'MLLM_data_3_v4',

    ]
    save_path = 'mllm_data_3_source_ids.json'
    ans_data = []
    for data_dir in data_dir_list:
        for subject in os.listdir(data_dir):
            if 'common_ratio_json_file' in subject:
                continue
            keyword = subject.split('_')[0].lower()
            subject_dir = os.path.join(data_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
            if subject_dir in data_dir_list:
                continue
            print(subject_dir)
            samples = os.listdir(subject_dir)
            now_data = iterate_mllm_data_3_thread(samples, subject_dir, keyword)
            ans_data.extend(now_data)
            print(f'ans_data num={len(ans_data)}')
    print(f'ans_data num={len(ans_data)}')
    save_json_file(save_path, ans_data)
    print(save_path)
    pass
