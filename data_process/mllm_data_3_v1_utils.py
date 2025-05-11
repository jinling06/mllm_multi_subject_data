"""
@Time: 2024/9/15 22:27
@Author: xujinlingbj
@File: mllm_data_3_v1_utils.py
"""
import json
import os
import re
import sys
from itertools import chain
import concurrent.futures
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor
from data_process.image_utils import open_svg, is_transparent, convert_transparent_to_white_background
from data_process.parse_html import parse_html_content
from data_process.remove_duplicate import filter_same_question
from data_process.utils import *


def parse_mllm_data_3_thread(samples, group_data, version_dir, save_path, index_prefix, keyword):
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

        # 过滤公式图片
        # formular_flag = False
        # for image_file in image_files:
        #     if judge_formula_image(image_file):
        #         formular_flag = True
        #         print(f'formular_flag={image_file}')
        #         break
        # if formular_flag:
        #     continue

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
        # print(raw_answer)

        text_list = re.split(r'【小题\d】', raw_answer)

        answer_list = []
        for x in text_list:
            x = format_answer(x)
            if contains_uppercase_abcd(x):
                answer_list.append(x)

        if len(answer_list) != len(html_data_list):
            print(f'答案个数和html解析的问题个数不一致')
            print(html_file_path)
            print(raw_answer)
            continue
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
        # print(html_data_list)
        # print(len(html_data_list))
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
                "id": f"mllm_data_3_v1_{index_prefix}_{index}",
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


def process_transparent_image(open_image, image_path):

    save_image_path = os.path.join("raw_data/", image_path)

    image_file_name = image_path.split("/")[-1]
    image_dir = "/".join(save_image_path.split("/")[:-1])
    image_file_name = ".".join(image_file_name.split(".")[:-1])
    new_save_image_path = os.path.join(image_dir, f"{image_file_name}_new.png")

    if is_transparent(open_image):
        print(f'转换透明图')
        open_image = convert_transparent_to_white_background(open_image)

        open_image.save(new_save_image_path)
        image_path = new_save_image_path.replace("raw_data/", "")
    return open_image, image_path


def filter_thread_data(data, image_prefix, rm_with_formula_image_data, processor, source_data_name,
                       filter_over_width_image):
    res_data = []
    for line in tqdm(data, total=len(data), desc='filter'):

        prompt = line['conversations'][0]['value']
        image_num = prompt.count('<image>')
        save_flag = True

        if 'image' in line:
            image = line['image']
            if isinstance(image, str):
                image_path = os.path.join(image_prefix, image)
                image_list_num = 1
                try:
                    img = Image.open(image_path)
                    if filter_over_width_image and judge_illegal_image(img):
                        print(f'过滤宽高不合法的图：width={img.width}, height={img.height}')
                        save_flag = False
                        continue
                    if rm_with_formula_image_data and judge_formula_image(image_path=image_path, open_image=img,
                                                                          source=source_data_name):
                        save_flag = False
                        print(f'过滤公式图')
                    if not save_flag:
                        continue
                    # 处理透明图
                    img, image = process_transparent_image(img, image)
                    line['image'] = image
                    _ = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]

                except Exception as e:
                    # print(json.dumps(line, ensure_ascii=False, indent=4))
                    save_flag = False
                    print(f'过滤error image')
                    print(e)
                    print(line['image'])

            else:
                if len(image) > 4:
                    save_flag = False
                    print(f'filter 图片个数大于4的数据')

                    continue

                image_list_num = len(image)
                res_image_list = []
                for image_name in image:
                    image_path = os.path.join(image_prefix, image_name)
                    try:
                        img = Image.open(image_path)
                        if filter_over_width_image and judge_illegal_image(img):
                            print(f'过滤宽高不合法的图：width={img.width}, height={img.height}')
                            save_flag = False
                            break
                        if rm_with_formula_image_data and judge_formula_image(image_path=image_path, open_image=img,
                                                                              source=source_data_name):
                            save_flag = False
                            print(f'过滤公式图')
                        if not save_flag:
                            break
                        # 处理透明图
                        img, image_name = process_transparent_image(img, image_name)

                        res_image_list.append(image_name)
                        _ = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                    except Exception as e:
                        print(e)
                        print(image_path)
                        print(f'过滤error image')
                        save_flag = False

                line['image'] = res_image_list
            if image_num != image_list_num:
                print('image token 和图片数量不相等', image_num, image_list_num)
                print(line['image'])
                save_flag = False
        if save_flag:
            res_data.append(line)
    return res_data


def filter_data(data_dir, save_path, rm_with_formula_image_data=True,
                source_data_name='mllm_3', filter_over_width_image=True):
    data = []
    if not os.path.isdir(data_dir):
        print(f'读入的是一个文件')
        data = load_json_file(data_dir)
    else:
        print(f'读入的是文件夹')
        for file in os.listdir(data_dir):
            data_path = os.path.join(data_dir, file)
            data.extend(load_json_file(data_path))
    print(f'原始数据量={len(data)}')
    # 根据question url 去重
    data = filter_same_question(data)
    # data = [
    #     x
    #     for x in data
    #     if x["common_info"]["common_ratio_base_test"] < 0.4 and x["common_info"]["common_ratio_base_custom"] < 0.4
    # ]
    # print(f"过滤重合测试集后数量={len(data)}")

    image_prefix = 'raw_data'
    processor = CLIPImageProcessor.from_pretrained("model/clip-vit-large-patch14-336")
    thread_num = 32
    res_data = []
    every_thread_data = get_thread_data_with_image_num(data, thread_num, print_log=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        results = [
            executor.submit(filter_thread_data, thread_data, image_prefix, rm_with_formula_image_data,
                            processor, source_data_name, filter_over_width_image) for thread_data in every_thread_data
        ]
        for future in concurrent.futures.as_completed(results):
            try:
                result = future.result()
                res_data.extend(result)
            except Exception as e:
                print(f"Thread generated an exception: {e}")

    save_json_file(save_path, res_data)
    print(f"res data num={len(res_data)}")
    print(f"save file={save_path}")


