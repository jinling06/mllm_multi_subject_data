"""
@Time: 2024/9/9 22:09
@Author: xujinlingbj
@File: preprare_mllm_data_1_v3.py
"""
import concurrent.futures
import json
import os
import random
import shutil
import sys
import re
import pandas as pd
from tqdm import tqdm

from data_process.image_utils import *
from data_process.mllm_data_3_v1_utils import filter_data
from data_process.prepare_mllm_data_v1 import judge_in_test_data
from data_process.utils import *


def replace_img_with_image(text):
    # 定义正则模式，匹配 <img ...> 标签
    # 正则表达式模式，用于匹配 <img> 标签
    img_pattern = r'<img(.*?)>'
    # 使用 re.sub 替换匹配到的内容为换行符
    replaced_text = re.sub(img_pattern, '\n<image>\n', text)

    return replaced_text



def extract_and_clean(text, pattern):
    # 定义匹配模式，用于提取"在线课程"和"解析"中间的字符串
    # pattern =

    # 使用re.search找到匹配的部分
    match = re.search(pattern, text, flags=re.DOTALL)  # 使用DOTALL标志以便匹配包括换行符在内的任意字符

    if match:
        # 提取匹配的内容
        extracted_string = match.group(1)

        # 用strip()移除字符串两端的换行符、空白和制表符
        cleaned_string = extracted_string.strip()

        # 移除字符串中的制表符和换行符'\r'， '\n'， '\t'
        cleaned_string = cleaned_string.replace('\n', '').replace('\r', '').replace('\t', '')

        return cleaned_string

    # 如果没有匹配结果，返回空字符串
    return ""

error_image = 0
image_num = 0


def process_chunk(chunk, tsv_data_path, now_image_save_dir):
    global error_image
    for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc='chunk'):
        image_name = row['image_name']
        if image_name.endswith('.wmf'):
            error_image += 1
            print(image_name)
            continue
        if '.' not in row['image_name']:
            print(f'error image name={image_name}')
            continue
        image_base64 = row['image_base64']

        save_image_path = os.path.join(now_image_save_dir, image_name)
        if os.path.exists(save_image_path):
            continue
        image = base64_to_image(image_base64)
        if is_transparent(image):
            image = convert_transparent_to_white_background(image)
        try:
            image.save(save_image_path)
        except:
            print(image_name, tsv_data_path)
            continue


def save_tsv_image(data_dir):
    global error_image, image_num

    num_threads = 32
    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject)
        keyword = subject.lower()
        if not os.path.isdir(subject_dir):
            print(f'error {subject_dir}')
            continue
        if subject in ['CHEMISTRY']:
            continue
        if '_' in subject:
            continue

        save_num = 0
        for file_name in os.listdir(subject_dir):
            file_path = os.path.join(subject_dir, file_name)
            if not file_path.endswith('.json'):
                # print(f'error file:{file_path}')
                continue
            print(file_path)
            tsv_data_path = os.path.join(subject_dir, file_name.split('.')[0] + '.tsv')
            print(tsv_data_path)
            tsv_data = pd.read_csv(tsv_data_path, sep='\t')

            image_num += len(tsv_data)
            now_image_save_dir = os.path.join(subject_dir, tsv_data_path.split('.')[0])
            # if os.path.exists(now_image_save_dir):
            #     shutil.rmtree(now_image_save_dir)

            os.makedirs(now_image_save_dir, exist_ok=True)
            # 切分数据为32个块
            chunks = np.array_split(tsv_data, num_threads)

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                for chunk in chunks:
                    executor.submit(process_chunk, chunk, tsv_data_path, now_image_save_dir)
                # for future in concurrent.futures.as_completed(futures):
                #     future.result()  # 处理抛出异常的情况

    print(f'原始图片数={image_num}, 出错图片数={error_image}')


def parse_thread_data(data, save_path, thread_prefix, keyword, image_name_to_full_path, current_class_test_data):
    index = 0
    res_data = []
    for line in tqdm(data, total=len(data), desc='make'):
        raw_question = line['raw_question']
        raw_answer = line['raw_answer']
        question = line['question']
        answer = line['answer']
        img_urls = line['img_urls']
        question_images = [url.split('/')[-1] for url in img_urls.get('question', [])]
        answer_images = [url.split('/')[-1] for url in img_urls.get('answer', [])]
        question_url = line['question_url']

        images = []
        save_flag = True

        if len(question_images) > 4:
            continue
        for image_name in question_images:
            if image_name not in image_name_to_full_path:
                save_flag = False
                break
            full_image_path = image_name_to_full_path[image_name]

            full_image_path = full_image_path.replace('raw_data/', '')
            images.append(full_image_path)
        if not save_flag:
            # print(line['img_urls'])
            # print(f'error image :{question_images}')

            continue

        image_token = "<image>" * len(images)
        answer = ''
        explanation = ''
        if contains_chinese(raw_answer):
            pattern_list = [f'【答案】(.*?)【解析】', f'在线课程(.*?)【解析】']
            # answer = extract_and_clean(raw_answer, r'在线课程(.*?)解析')
            for pattern in pattern_list:
                answer = extract_and_clean(raw_answer, pattern)
                if answer != '':
                    explain_index = raw_answer.find('【解析】')
                    end_index = raw_answer.find('考点')
                    if end_index != -1:
                        explanation = raw_answer[explain_index:end_index]
                    else:
                        explanation = raw_answer[explain_index:]
                    explanation = explanation.replace('【解析】', '').replace('\n', '').replace('\r', '').replace('\t', '')
                    explanation = explanation.replace('试题分析：', '').replace('解：', '').replace('考点', '')

                    break
        answer = format_answer(answer)
        if check_illegal_answer(answer):
            # print(f'illegal answer:{raw_answer}')
            continue
        if len(explanation) < 10:
            continue

        question = replace_img_with_image(question)
        if question.count('<image>') != len(images):
            print(f'问题中的图片数量和给的图片数量不相等', line['question'])
            print(f'替换image字符后的question={question}')
            print(json.dumps(line, ensure_ascii=False, indent=4))
            print(question_images)
            raise ValueError
        common_info = judge_in_test_data(current_class_test_data, question)
        question = insert_abcd_in_item(question)
        prompt = f"下面是一道##需要输出解析的##【{subject_dic[keyword]}】题，根据图示，回答下面的问题。\n问题是：{question}\n你的答案是："

        conversations = [
            {
                'from': 'human',
                'value': prompt,
            },
            {
                'from': 'gpt',
                'value': f"解析是：{explanation}\n答案是：" + answer,
            }
        ]
        temp_dic = {
            'id': f'mllm_data_1_v3_{thread_prefix}_{index}',
            'keyword': keyword,
            'keyword_id': keyword2id[keyword],
            'conversations': conversations,
            'question_url': question_url,
            'raw_answer': raw_answer,
            'common_info': common_info,
        }
        if len(images) > 0:
            temp_dic['image'] = images
        index += 1
        res_data.append(temp_dic)


    print(f'res_data num={len(res_data)}')
    save_json_file(save_path, res_data)
    print(f'save file={save_path}')


def parse_mllm_data_1_v3(data_dir, json_save_dir):
    if os.path.exists(json_save_dir):
        shutil.rmtree(json_save_dir)
    os.makedirs(json_save_dir, exist_ok=True)
    num = 0
    test_data_path = "raw_data/A_test/questions.json"
    group_data = group_raw_test_data(test_data_path)
    print(group_data.keys())
    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject)
        keyword = subject.lower()
        if not os.path.isdir(subject_dir):
            print(f'error {subject_dir}')
            continue
        if subject not in ['CHEMISTRY']:
            continue
        if subject == json_save_dir.split('/')[-1]:
            continue
        if '_' in subject:
            continue
        save_num = 0
        for file_name in os.listdir(subject_dir):
            file_path = os.path.join(subject_dir, file_name)
            if not file_path.endswith('.json'):
                # print(f'error file:{file_path}')
                continue

            print(file_path)
            data = load_json_file(file_path)
            file_prefix = file_name.split('.')[0]
            tsv_data_path = os.path.join(subject_dir, file_prefix+'.tsv')
            print(tsv_data_path)
            tsv_data = pd.read_csv(tsv_data_path, sep='\t')
            now_image_save_dir = os.path.join(subject_dir, file_prefix)
            image_name_to_full_path = defaultdict(str)

            for idd, row in tqdm(tsv_data.iterrows(), total=len(tsv_data), desc='tsv'):

                image_name = row['image_name']

                if '.' not in row['image_name']:
                    # print(f'error image name={image_name}')
                    continue
                if image_name.endswith('.wmf'):

                    continue
                image_name = '_'.join(image_name.split('_')[4:])

                now_save_image_path = os.path.join(now_image_save_dir, row['image_name'])

                if os.path.exists(now_save_image_path):
                    image_name_to_full_path[image_name] = now_save_image_path
            print(len(image_name_to_full_path.keys()))
            random.shuffle(data)
            thread_num = 32
            every_thread_data = get_thread_data(data, thread_num)
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
                for thread_id, thread_data in enumerate(every_thread_data):
                    thread_prefix = file_prefix+'_'+str(thread_id)
                    now_thread_save_path = os.path.join(json_save_dir, thread_prefix+'.json')
                    executor.submit(parse_thread_data, thread_data, now_thread_save_path, thread_prefix,
                                    keyword, image_name_to_full_path, group_data[keyword])
            save_num += 1

            # if save_num > 2:
            #     break



if __name__ == '__main__':
    data_dir = 'MLLM_data_1_v3_tsv'
    save_path = f'{data_dir}/train_0915_with_explain_chemistry.json'
    only_image_save_path = f'{data_dir}/train_0915_with_explain_with_image_chemistry.json'
    common_ration_json_save_dir = os.path.join(data_dir, 'common_ration_json_0915_with_explain_chemistry')
    # save_tsv_image(data_dir)
    parse_mllm_data_1_v3(data_dir, common_ration_json_save_dir)
    filter_data(common_ration_json_save_dir, save_path)
    filter_with_image(save_path, only_image_save_path)
    pass
