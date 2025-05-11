"""
@Time: 2024/8/28 20:57
@Author: xujinlingbj
@File: prepare_mllm_data_v1.py
"""
import os
import random
import shutil
import sys
import threading
import multiprocessing
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_process.image_utils import *
from data_process.mllm_data_3_v1_utils import filter_data
from data_process.utils import *


def get_json_info(json_path):
    data = load_json_file(json_path)
    query = data["text_1"]
    query = query.replace("\n", "").replace("\r", "")
    text_2 = data["text_2"]
    text_2 = text_2.replace("\n", "").replace("\r", "")
    question_url = data["question_url"]

    answers = parse_answer_from_custom_data(text_2)
    if has_other_than_uppercase(answers):
        answers = split_multi_question_text(answers)
        question_list = split_multi_question_text(query)
    else:
        answers = [answers]
        question_list = [query]
    res_dict_list = []

    for question, answer in zip(question_list, answers):
        if answer == "":
            print(f'答案为空: {text_2}\n'
                  f'question:{query}')
            continue

        if check_illegal_answer(answer):
            print(f'非法答案: {text_2}\n'
                  f'question:{query}')
            continue
        # 去掉填空题
        if has_other_than_uppercase(answer):
            continue
        answer = format_answer(answer)
        if len(answer.split("、")) > 4:
            print(f'答案个数超过4: {text_2}\n'
                  f'question:{query}')
            continue
        question = insert_abcd_in_item(question)
        prompt = f"问题是：{question}。\n你的答案是："
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
            "query": query,
            "answer": text_2,
            "conversations": conversations,
            "question_url": question_url+'-'+str(len(res_dict_list)),
        }
        res_dict_list.append(temp_dic)

    return res_dict_list



def process_image_file(image_data, image_dir, json_dir, source_name, category, keyword,
                       now_thread_save_path, save_prefix):
    index = 0
    res_data = []
    for image_file in tqdm(image_data, total=len(image_data), desc="image"):
        image_path = os.path.join(image_dir, image_file)
        json_name = image_file.replace("None", "img")
        json_path = os.path.join(json_dir, json_name + ".json")

        if not os.path.exists(json_path):
            # print(f"miss {json_path}")
            continue

        try:
            _ = Image.open(image_path)
        except Exception as e:
            print(e)
            continue

        images = image_path.replace("raw_data/", "")
        image_token = "<image>"
        temp_dic_list = get_json_info(json_path)

        if len(temp_dic_list) == 0:
            continue
        for temp_dic in temp_dic_list:

            temp_dic["id"] = f"{save_prefix}_{index}"
            temp_dic["keyword"] = keyword
            temp_dic['keyword_id'] = keyword2id[keyword]
            temp_dic["image"] = images
            temp_dic["conversations"][0]["value"] = (
                f"{image_token}\n下面是一道【{subject_dic[keyword]}】题，\n根据图示，回答下面的问题。\n"
                + temp_dic["conversations"][0]["value"]
            )
            res_data.append(temp_dic)
            index += 1

    save_json_file(now_thread_save_path, res_data)

#
# def process_json_file(json_data, json_dir, source_name, category, group_data, keyword):
#     global res_data, index
#     for json_name in tqdm(json_data, total=len(json_data), desc="json"):
#         json_path = os.path.join(json_dir, json_name)
#
#         temp_dic_list = get_json_info(json_path, group_data[keyword])
#         if len(temp_dic_list) == 0:
#             continue
#         for temp_dic in temp_dic_list:
#             data_lock.acquire()
#             temp_dic["id"] = f"{source_name}_{category}_{index}"
#             temp_dic["keyword"] = keyword
#             temp_dic["conversations"][0]["value"] = (
#                 f"下面是一道【{subject_dic[keyword]}】题，根据问题描述，回答下面的问题。\n" + temp_dic["conversations"][0]["value"]
#             )
#             res_data.append(temp_dic)
#             index += 1
#             data_lock.release()


def parse_mllm_data(data_dir_list, json_save_dir):
    if os.path.exists(json_save_dir):
        shutil.rmtree(json_save_dir)
    os.makedirs(json_save_dir, exist_ok=True)
    json_save_dir_name = json_save_dir.split('/')[-1]
    num_threads = 32
    # test_data_path = "raw_data/A_test/questions.json"
    # group_data = group_raw_test_data(test_data_path)
    # print(group_data.keys())

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for data_dir in data_dir_list:
            category_files = os.listdir(data_dir)
            source_name = data_dir.split("/")[-1]

            for category in category_files:
                if 'common_ratio_json' in category:
                    print(f'filter ={category}')
                    continue
                category_dir = os.path.join(data_dir, category)
                if not os.path.isdir(category_dir):
                    continue
                print(category_dir)
                keyword = category.split("_")[0].lower()

                image_dir = os.path.join(category_dir, "images")
                image_files = os.listdir(image_dir)

                json_dir = os.path.join(data_dir, category, "meta_data")

                image_thread_data = get_thread_data(image_files, num_threads=num_threads)
                for thread_id, image_data in tqdm(enumerate(image_thread_data), total=len(image_files), desc=f"{category}"):
                    thread_prefix = source_name + '_' +category + '_' + str(thread_id)
                    now_thread_save_path = os.path.join(json_save_dir, thread_prefix + '.json')
                    print(now_thread_save_path)
                    executor.submit(
                        process_image_file, image_data, image_dir, json_dir, source_name, category,
                        keyword, now_thread_save_path, thread_prefix
                    )

                # json_list = os.listdir(json_dir)
                # json_list = [x for x in json_list if "no_img" in x]
                # json_thread_data = get_thread_data(json_list, num_threads=num_threads)
                #
                # for json_data in tqdm(json_thread_data, desc="json"):
                #     executor.submit(process_json_file, json_data, json_dir, source_name, category, group_data, keyword)
                # break


if __name__ == "__main__":
    data_dir = "MLLM_data_1_v1"
    with_common_ratio_save_dir = f"{data_dir}/common_ratio_json"
    parse_mllm_data(
        data_dir_list=[data_dir, "MLLM_data_1_v2"],
        json_save_dir=with_common_ratio_save_dir,
    )
    filter_save_path = f"{data_dir}/train_1031.json"
    filter_data(with_common_ratio_save_dir, filter_save_path, source_data_name='mllm_1')

