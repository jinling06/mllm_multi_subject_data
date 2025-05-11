"""
@Time: 2024/9/10 22:07
@Author: xujinlingbj
@File: prepare_chinese_data.py
"""
import json
import os
import random
import concurrent.futures
import re
import shutil
import sys
from itertools import chain

from data_process.image_utils import *
from data_process.mllm_data_3_v1_utils import filter_data
from data_process.parse_html import parse_html_content
from data_process.prepare_mllm_data_v1 import judge_in_test_data
from data_process.utils import *


def parse_thread(samples, group_data, version_dir, save_path, index_prefix, keyword):
    error_num = 0
    all_num = 0
    index = 0

    res_data = []
    for sample_name in tqdm(samples, total=len(samples), desc="sample"):
        sample_dir = os.path.join(version_dir, sample_name)
        if not os.path.isdir(sample_dir):
            print(f'error sample_dir={sample_dir}')
            continue
        image_files = []
        html_file_path = ""
        answer_file_path = ""
        parse_file_path = ""
        # if sample_name != 'save_sample_40002067_url_zsd45412_page_156':
        #     continue
        # print(os.listdir(sample_dir))
        source_data_id = ''
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
                        if not os.path.exists(png_file_path):
                            image = open_svg(file_path)
                            image.save(png_file_path)
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

        image_files = [x.replace("raw_data/", "") for x in image_files]

        # print(all_num)
        # all_num += 1
        # print(html_file_path)
        #
        # print(image_files)

        html_data_list, paragraph_image_list, question_image_list = parse_html_content(html_file_path)

        contain_image_num = len(paragraph_image_list) + len(list(chain.from_iterable(question_image_list)))
        # print(f'contain_image_num={contain_image_num}, image num={len(image_files)}')
        if contain_image_num > len(image_files):
            continue
        if len(html_data_list) == 0:
            print(f'html为空：{html_file_path}')
            continue
        if answer_file_path == '':
            print(f'miss answer file={html_file_path}')
            continue

        answer_data = load_json_file(answer_file_path)[0]

        try:
            answer_data = parse_ocr_text(answer_data)
        except Exception as e:
            print(e)
            print(f'answer 文件错误：{answer_data}')
            continue
        text_list = [x["text"] for x in answer_data]
        raw_answer = "".join(text_list)
        raw_answer = raw_answer[raw_answer.find('【小题'):]

        text_list = re.split(r'【小题\d】', raw_answer)

        # text_list = [x for x in text_list if len(x) > 0 and x[0].isupper()]
        answer_list = []
        for x in text_list:
            x = format_answer(x)
            if contains_uppercase_abcd(x):
                answer_list.append(x)
        # print(answer_list)

        # print(len(html_data_list))
        min_len = min(len(answer_list), len(html_data_list))
        answer_list = answer_list[:min_len]
        html_data_list = html_data_list[:min_len]
        # if len(answer_list) < len(html_data_list):
        #     print(f'答案个数小于html解析的问题个数')
        #     print(html_file_path)
        #     print(raw_answer)
        #     continue

        # if len(answer_list) < 2:
        #     continue

        image_name_to_file_path = defaultdict(str)
        for file_path in image_files:
            file_name = '_'.join(file_path.split('/')[-1].split('_')[2:])
            file_name = file_name.split('.')[0]
            image_name_to_file_path[file_name] = file_path
        # print(paragraph_image_list)
        # print(question_image_list)
        # print(image_name_to_file_path)
        # print(len(question_image_list), len(paragraph_image_list))
        #
        # print(source_data_id)
        # sys.exit(0)
        for i in range(len(html_data_list)):

            question = html_data_list[i]
            answer = answer_list[i]

            common_info = judge_in_test_data(group_data[keyword], question)
            prompt = f"下面是一道【{subject_dic[keyword]}】题，根据问题描述，回答下面的问题。\n{question}"
            image_count = prompt.count('<image>')
            now_image_names = []
            now_image_names.extend(paragraph_image_list)

            if i < len(question_image_list):
                now_image_names.extend(question_image_list[i])
            now_image_names = [x.split('_')[-1] for x in now_image_names]
            # if len(now_image_names) > 4:
            #     print(f'图片数量大于4：{html_file_path}')
            #     continue
            if image_count != len(now_image_names):
                print(f'image token数量和图片数量不相等:{html_file_path}')
                print(prompt)
                print(image_count)
                print(now_image_names)
                continue
            save_flag = True

            now_images = []
            for x in now_image_names:
                if x not in image_name_to_file_path:
                    save_flag = False
                    print(f'图片不在保存的文件中{x}, {html_file_path}')
                    break
                if not os.path.exists(os.path.join('raw_data', image_name_to_file_path[x])):
                    save_flag = False
                    print(f'图片不存在:{image_name_to_file_path[x]}, {html_file_path}')
                    break
                now_images.append(image_name_to_file_path[x])

            if not save_flag:
                continue
            if check_illegal_answer(answer):
                continue
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

            temp_dic = {
                "id": f"{index_prefix}_{index}",
                "category": keyword,
                "keyword": keyword,
                "keyword_id": keyword2id[keyword],
                "conversations": conversations,
                "raw_answer": raw_answer,
                "common_info": common_info,
                "source_id": f'{source_data_id}_{i}',
                'html_path': html_file_path,
            }
            if len(now_images) > 0:
                temp_dic["image"] = now_images
            index += 1
            # print(json.dumps(temp_dic, ensure_ascii=False, indent=4))
            res_data.append(temp_dic)
        # if index > 20:
        #     break

    print(f"res_data num={len(res_data)}")
    save_json_file(save_path, res_data)
    print(f"save file={save_path}")


def parse_chinese(data_dir, json_save_dir):
    if os.path.exists(json_save_dir):
        shutil.rmtree(json_save_dir)

    os.makedirs(json_save_dir, exist_ok=True)
    test_data_path = "raw_data/A_test/questions.json"
    group_data = group_raw_test_data(test_data_path)
    print(group_data.keys())
    thread_num = 32
    keyword = "chinese"
    for version_name in os.listdir(data_dir):
        version_dir = os.path.join(data_dir, version_name)
        if not os.path.isdir(version_dir):
            continue
        if 'common_ratio_json_file' in version_name:
            continue
        samples = os.listdir(version_dir)
        every_thread_data = get_thread_data(samples, thread_num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
            for thread_id, thread_data in enumerate(every_thread_data):
                thread_prefix = version_name + '_' + str(thread_id)
                now_thread_save_path = os.path.join(json_save_dir, thread_prefix + '.json')
                executor.submit(parse_thread, thread_data, group_data, version_dir, now_thread_save_path,
                                thread_prefix, keyword)


if __name__ == "__main__":
    data_dir = "MLLM_data_3_v1/CHINESE"
    save_path = f"{data_dir}/train_1110.json"
    with_common_ratio_save_path = f"{data_dir}/common_ratio_json_file_with_formula"
    only_image_save_path = f'{data_dir}/train_1110_with_image.json'

    # parse_chinese(data_dir, with_common_ratio_save_path)

    filter_data(with_common_ratio_save_path, save_path, source_data_name='mllm_3', filter_over_width_image=False)
    filter_with_image(save_path, only_image_save_path)

    pass
