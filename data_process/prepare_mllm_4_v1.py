"""
@Time: 2024/10/31 22:42
@Author: xujinlingbj
@File: prepare_mllm_4_v1.py
"""
import json
import os.path
import random
import shutil
import sys
import concurrent.futures
from bs4 import BeautifulSoup
from pylatexenc.latex2text import LatexNodes2Text

from data_process.mllm_data_3_v1_utils import filter_data
from data_process.utils import *

def parse_html(html_text):

    soup = BeautifulSoup(html_text, 'lxml')
    # 提取所有的<img>标签
    img_tags = soup.find_all('img')

    # 获取所有图片链接
    img_urls = [img['src'] for img in img_tags]
    # 提取问题
    question_div = soup.find('div', class_='pt1')
    question_text = question_div.get_text(separator=' ', strip=True)
    question_text = " ".join(question_text.split()[2:])  # 去掉前面的题号和年份信息

    # 提取选项
    options_table = soup.find('table', class_='quesborder')
    options = options_table.find_all('td', class_='selectoption')

    def get_latex_part(element):
        if not element:
            return ''

        latex_parts = []

        # Recursively process child elements
        for child in element.children:

            if 'mfrac' in child.get('class', []):
                numer = ''.join([get_latex_part(n) for n in child.find_all('div', class_='fracZi')])
                denom = ''.join([get_latex_part(d) for d in child.find_all('div', class_='fracMu')])

                latex_parts.append(f"\\frac{{{numer}}}{{{denom}}}")
            elif 'msqrt' in child.get('class', []):

                latex_parts.append(get_latex_part(child))
            elif 'msqrtBox' in child.get('class', []):
                latex_parts.append(f"\\sqrt{{{get_latex_part(child)}}}")

            elif 'msubsup' in child.get('class', []):
                latex_parts.append(get_latex_part(child))

            elif 'msub' in child.get('class', []):
                base = child.get_text()
                latex_parts.append(f"_{{{base}}}")

            elif 'msup' in child.get('class', []):

                base = child.get_text()
                latex_parts.append(f"^{{{base}}}")

            else:
                if child.get_text() != '√':

                    latex_parts.append(child.get_text())

        return ''.join(latex_parts)

    def extract_latex(math_div):
        latex_parts = []
        processed_parts = set()

        def mark_processed(element):
            """递归地标记元素及其子元素为已处理"""
            if element:
                processed_parts.add(id(element))
                for child in element.find_all(recursive=False):  # 仅处理直接子元素
                    mark_processed(child)

        for part in math_div.children:
            latex_parts.append(get_latex_part(part))

        for part in math_div.find_all('sub'):
            base = part.previous_sibling if part.previous_sibling else ""
            base_text = base.get_text() if base else ""
            subscript = part.get_text()
            latex_parts.append(f"{base_text}_{{{subscript}}}")

        return ''.join(latex_parts)

    options_text = []
    for option in options:
        option_text = option.get_text(separator=' ', strip=True)  # 提取普通文本内容
        # 在文本中寻找公式
        formula_divs = option.find_all('div', class_='MathJye')
        for formula_div in formula_divs:
            latex_formula = extract_latex(formula_div)

            option_text = option_text.replace(formula_div.get_text(separator=' ', strip=True), latex_formula)

        # 替换直接在文本中的 <sub>标签
        for sub in option.find_all('sub'):
            sub_text = sub.get_text()
            base_text = sub.previous_sibling if sub.previous_sibling else ""
            base_text = base_text.get_text() if base_text else ""
            option_text = option_text.replace(f"{base_text} {sub_text}", f"{base_text}_{{{sub_text}}}")
        options_text.append(option_text)

    # 打印问题和选项
    # print(html_text)
    # print("Question:", question_text)
    # print("Options:")
    # for i, option in enumerate(options_text):
    #     print(f"{chr(65 + i)}. {option}")
    return question_text, options_text, img_urls


def get_html_text(html_text):
    soup = BeautifulSoup(html_text, 'lxml')
    plain_text = soup.get_text()
    return plain_text


def parse_mllm_data_thread(data, save_path, keyword, index_prefix, image_save_dir):
    res_data = []
    index = 0
    for line in tqdm(data, total=len(data), desc='parse'):
        key = line['key']
        html_content = line['html_content']

        raw_question_text, raw_options_text, img_urls = parse_html(html_content)
        if len(img_urls) != 1:
            print(f'非法图片：{img_urls}')
            continue
        images = []
        save_flag = True
        for image_url in img_urls:
            index = image_url.find('net/') + 4
            image_save_name = f'{index_prefix}_{index}_' + image_url[index:].replace('/', '_')

            image_save_path = os.path.join(image_save_dir, image_save_name)

            if not download_image(image_url, image_save_path):
                print(f'下载图片失败')
                save_flag = True
                break
            image_path = image_save_path.replace('raw_data/', '')
            images.append(image_path)

        if not save_flag:
            continue
        raw_options_text = '\n'.join(raw_options_text)
        question_text = LatexNodes2Text().latex_to_text(raw_question_text)
        options_text = LatexNodes2Text().latex_to_text(raw_options_text)

        image_token = '<image>'*len(img_urls)
        prompt = f'{image_token}\n根据问题描述，回答下面的问题。\n问题是：{question_text} \n选项是：\n{options_text} \n你的答案是：'
        prompt = prompt.replace(' ', '')
        answer = key.split('.')[0].split('_')[-1].strip()
        answer = format_answer(answer)
        if check_illegal_answer(answer):
            print(f'过滤非法答案：{key}')

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
            "id": f"mllm_data_4_v1_{index_prefix}_{index}",
            "keyword": keyword,
            "keyword_id": keyword2id[keyword],
            "conversations": conversations,
            'raw_question_text': raw_question_text,
            'raw_options_text': raw_options_text,
            'html_content': html_content,
            'html_name': key,
        }
        if len(images) > 0:
            temp_dic['image'] = images
        # print(key)
        # print(prompt)
        # print(json.dumps(temp_dic, ensure_ascii=False, indent=4))
        # sys.exit(0)
        index += 1
        res_data.append(temp_dic)
        # if len(res_data) > 10:
        #     sys.exit(0)
    save_json_file(save_path, res_data)


def parse_data(data_path, json_save_dir, subject='PHYSICS'):
    data = load_json_file(data_path)
    print(f'raw data={len(data)}')
    if os.path.exists(json_save_dir):
        shutil.rmtree(json_save_dir)
    os.makedirs(json_save_dir)
    data = [{'key': key, 'html_content': value} for key, value in data.items()]
    print(f'raw data={len(data)}')
    random.shuffle(data)
    subject = subject.lower()
    thread_num = 32
    every_thread_data = get_thread_data(data, thread_num, print_log=True)
    image_save_dir = os.path.join(os.path.dirname(data_path), f'{subject}_images')
    print(image_save_dir)
    if os.path.exists(image_save_dir):
        shutil.rmtree(image_save_dir)
    os.makedirs(image_save_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        for thread_id, thread_data in enumerate(every_thread_data):
            thread_prefix = subject + '_' + str(thread_id)
            now_thread_save_path = os.path.join(json_save_dir, thread_prefix + '.json')
            print(now_thread_save_path)
            executor.submit(parse_mllm_data_thread, thread_data, now_thread_save_path, subject, thread_prefix, image_save_dir)


if __name__ == '__main__':

    data_dir = 'MLLM_data_4_v1'
    data_path = f'{data_dir}/PHYSICS_v1_html_ques.json'
    common_save_path = f'{data_dir}/common_ratio_json_file'
    save_path = f'{data_dir}/train_1101.json'
    only_image_save_path = f'{data_dir}/train_1101_filter_with_image.json'
    parse_data(data_path, common_save_path, subject='PHYSICS')
    filter_data(common_save_path, save_path, source_data_name='mllm_1')
    filter_with_image(save_path, only_image_save_path)