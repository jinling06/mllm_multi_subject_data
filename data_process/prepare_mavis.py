"""
@Time: 2024/8/28 13:12
@Author: xujinlingbj
@File: prepare_mavis.py
"""
import os
import random
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

from data_process.prompt_data import english_prompt_list
from data_process.utils import *


def parse_mavis_caption_data(data_path, save_path):
    data = load_json_file(data_path)
    print(f"data num ={len(data)}")
    res_data = []
    index = 0

    for line in tqdm(data, desc="parse"):
        image = line["image"]
        idd = image.find("MAVIS_Caption")
        if idd == -1:
            print(line)
            sys.exit(0)
        line["raw_id"] = line["id"]

        image = image[idd:]
        image = os.path.join("MAVIS", image)
        save_image_path = os.path.join("raw_data/", image)
        if not os.path.exists(save_image_path):
            print(f"image not find: {save_image_path}")
            continue
        line["id"] = f"mavis_{index}"
        line["image"] = image
        index += 1
        res_data.append(line)
    print(f"res_data num={len(res_data)}")
    save_json_file(save_path, res_data)
    print(f"save file={save_path}")


def parse_mavis_instruct_data(data_path_list, save_path):
    """
    选择题 prompt
    <image>\nAccording to the question shown in the image, please first perform reasoning, then finally select the right answer from the choices, e.g., Answer: xxx.\nQuestion: {}\nChoices:{}

    问答题 prompt
    <image>\nAccording to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., The answer is xxx\nQuestion: {}
    """
    all_subject = []
    index = 0
    res_data = []
    saved_question = []
    cnt = 0
    keyword2id = get_keyword2id()
    for data_path in data_path_list:
        print(data_path)
        data = load_json_file(data_path)
        image_prefix = data_path[data_path.find("MAVIS") :]
        image_prefix = "/".join(image_prefix.split("/")[:-1])
        print(f"data num ={len(data)}")
        for line in tqdm(data, desc="parse"):
            image = line["image"]
            subject = line["subject"]
            image = image.replace("RuleBaseGeo_For_Vision_Dominant", "rule_base_geo_vision_dom")
            if "rule_base_geo_vision_dom" in image:
                image = image[image.find("rule_base_geo_vision_dom") :]
            image = os.path.join(image_prefix, image)

            save_image_path = os.path.join("raw_data/", image)
            if not os.path.exists(save_image_path):
                # print(f"image not find: {save_image_path}")
                # print(line)
                # sys.exit(0)
                continue
            conversations = line["conversations"]
            # if conversations[0]['value'] in saved_question:
            #     continue
            # saved_question.append(conversations[0]['value'])
            # 原始数据中
            # 问答和填空 用的 The answer is
            # 选择用的 Answer:
            question_type = line['question_type']
            if question_type == 'open-ended':
                cnt += 1
            answer = (
                conversations[1]["value"].split("\n")[-1].replace("Answer:", "").replace("The answer is", "").strip()
            )
            if 'Answer:' in conversations[1]["value"] and len(answer) > 1:
                answer = answer.split('.')[0].strip()
            if len(answer) == 0:
                # print(conversations[1]["value"])
                # sys.exit(0)
                continue
            prompt = (
                conversations[0]["value"]
                .replace("first conduct reasoning, and then ", "")
                .replace("first perform reasoning, then finally ", "")
                .replace(", e.g., Answer: xxx", "").replace('e.g., The answer is xxx', '')
            )
            answer = "The answer is: " + answer
            if len(answer) > 20:
                continue
            # answer = answer.strip()
            conversations[0]["value"] = prompt
            conversations[1]["value"] = answer
            line["id"] = f"mavis_instruct_{index}"
            line["image"] = image
            line['keyword'] = 'math'
            line['keyword_id'] = keyword2id['math']
            line['subject'] = subject
            line['system'] = random.choice(english_prompt_list)
            index += 1
            all_subject.append(subject)
            res_data.append(line)

    print(Counter(all_subject))
    print(cnt)
    print(f"res_data num={len(res_data)}")
    save_json_file(save_path, res_data)
    print(f"save file={save_path}")


if __name__ == "__main__":
    save_path = sys.argv[1]
    parse_mavis_instruct_data(
        data_path_list=[
            "MAVIS/Caption_to_QA/Function_Caption_to_Question.json",
            "MAVIS/Caption_to_QA/Geometry_Caption_to_Question.json",
            "MAVIS/Existing_Dataset_Augment/Existing_Dataset_Augment.json",
            ],
        save_path=save_path,
    )
    pass
