"""
@Time: 2024/8/27 22:35
@Author: xujinlingbj
@File: prepare_SEED-Bench-H.py
"""
import json
import os
import sys
from tqdm import tqdm
from data_process.utils import save_json_file, load_json_file


def parse_seed_bench_h(data_path, save_path):
    """
    自然图片 QA
    {'Scene Understanding': 1, 'Instance Identity': 2, 'Instance Attributes': 3,
    'Instance Location': 4, 'Instances Counting': 5, 'Spatial Relation': 6,
    'Instance Interaction': 7, 'Visual Reasoning': 8, 'Text Understanding': 9,
    'Celebrity Recognition': 10, 'Landmark Recognition': 11, 'Chart Understanding': 12,
    'Visual Referring Expression': 13, 'Science Knowledge': 14, 'Emotion Recognition': 15,
    'Visual Mathematics': 16, 'Difference Spotting': 17, 'Meme Comprehension': 18,
    'Global Video Understanding': 19, 'Action Recognition': 20, 'Action Prediction': 21,
    'Procedure Understanding': 22, 'In-Context Captioning': 23, 'Interleaved Image-Text Analysis': 24,
    'Text-to-Image Generation': 25, 'Next Image Prediction': 26, 'Text-Image Creation': 27,
    'Few-shot Segmentation': 28, 'Few-shot Keypoint': 29, 'Few-shot Depth': 30,
    'Few-shot Object Detection': 31, 'Image to Latex': 32, 'Text-Rich Visual Comprehension': 33}

    ['Single-Image & Text Comprehension', 'Video & Text Comprehension',
    'Multiple-Images & Text Comprehension', 'Interleaved Image & Text Comprehension',
    'Image Generation', 'Image & Text Generation']
    """
    data = load_json_file(data_path)
    res_data = []
    index = 0
    question_type = data["question_type"]
    questions = data["questions"]
    print(question_type)
    # print(f"questions num={len(questions)}")

    for info in tqdm(questions, total=len(questions), desc='make'):
        answer = info["answer"]
        option_id = ["choice_a", "choice_b", "choice_c", "choice_d"]
        option_id_2 = ["choice_A", "choice_B", "choice_C", "choice_D"]
        data_source = info["data_source"]
        data_id = info["data_id"]
        question = info["question"]
        question_id = info["question_id"]
        data_type = info['data_type']
        subpart = info['subpart']
        if subpart not in ['Single-Image & Text Comprehension']:
            continue
        image = data_id

        if data_source == 'cc3m':
            image = os.path.join('cc3m-image', image)
        elif 'task' in data_id:
            image = os.path.join('SEED-Bench-2-image', image)
        elif 'latex_code' not in data_id and 'text_rich' not in data_id:
            image = os.path.join('SEED-Bench-H-data', image)

        image = os.path.join('llava_sft/math_data/SEED-Bench-H', image)
        save_image_path = os.path.join('raw_data/', image)
        if not os.path.exists(save_image_path):
            print(f'image not find={save_image_path}')
            sys.exit(0)
            continue
        choose_text = ''

        st_a = 'A'
        if option_id[0] in info:
            for x in option_id:
                choose_text += '\n' + st_a + '、' + info[x]
                st_a = chr(ord(st_a) + 1)
        elif option_id_2[0] in info:
            for x in option_id_2:
                choose_text += '\n' + st_a + '、' + info[x]
                st_a = chr(ord(st_a) + 1)
        else:
            print(info)
            sys.exit(0)

        if choose_text != '':
            choose_text = 'Choices:' + choose_text
        hint = "This is a multiple-choice question, choose the answer from the question description"
        if choose_text == '':
            hint = "This is a quiz question, please give your answer directly"
        image_token = "<image>"
        prompt = f"{image_token}\nBased on the diagram, answer the following questions.\nThe question is: {question}。{choose_text}\n" f"Please note:{hint}."

        conversations = [
            {
                'from': 'human',
                'value': prompt,
            },
            {
                'from': 'gpt',
                'value': "The answer is: " + answer,
            }
        ]
        temp_dic = {
            'id': f'seed_bench_h_{question_id}_{index}',

            'images': image,
            'conversations': conversations,

        }
        index += 1

        res_data.append(temp_dic)

    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


if __name__ == "__main__":
    # 19k
    data_path = "SEED-Bench-H/SEED-Bench-H.json"
    save_path = "SEED-Bench-H/train_0827.json"
    parse_seed_bench_h(data_path, save_path)
    pass
