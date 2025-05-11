"""
@Time: 2024/10/20 16:12
@Author: xujinlingbj
@File: clean_train_data.py
"""
import os.path
import sys

from PIL import ImageDraw, Image, ImageFont
import concurrent.futures
from data_process.image_utils import is_transparent, convert_transparent_to_white_background
from data_process.utils import *
from data_process.prepare_only_text_data import check_answer, reformat_prompt

# 英文到中文标点符号的映射
punctuation_map = {
    # '.': '。',
    ',': '，',
    '?': '？',
    '!': '！',
    ':': '：',
    ';': '；',
    '-': '－',
    '(': '（',
    ')': '）',
    '[': '【',
    ']': '】',
    '{': '｛',
    '}': '｝',
    '"': '“',
    "'": '‘',
}


def english_to_chinese_punctuation(text):
    # 遍历字符串中的每个字符
    new_text = ""
    for char in text:
        # 如果字符是英文标点符号，则替换为中文标点符号
        if char in punctuation_map:
            new_text += punctuation_map[char]
        else:
            new_text += char
    return new_text


def clean_question(question):
    if "下表" in question or "如表" in question or "表格" in question or "此表" in question or "该表" in question or "这表" in question or "高.考.资.源.网" in question or "\n问题是：选项是" in question:
        return None
    if question.count("<image>") == 3:
        return None

    if (question.count("\nA") == 1 and question.count("\nB") == 1 and question.count("\nC") == 1 and question.count(
            "\nD") == 1) or question.count("<image>") >= 4:

        if "选项是:\nA" not in question and "选项是：\nA" not in question:
            question = question.replace("\nA.", "\n选项是：\nA.")
            question = question.replace("\nA、", "\n选项是：\nA、")
            question = question.replace("\nA．", "\n选项是：\nA．")

        # if "选项是:\n" in question:
        #     question = question.replace("选项是:\n", "\n选项是：\n")
        while ' ' in question:
            question = question.replace(" ", "")
        while '\u3000' in question:
            question = question.replace("\u3000", "")
        while '（ ' in question:
            question = question.replace('（ ', "（")
        while ' ）' in question:
            question = question.replace(' ）', "）")
        question = english_to_chinese_punctuation(question)
        return question
    else:
        return None


def add_option_to_image(image, text):
    # 在图像底部添加20像素的空白
    width, height = image.size
    # 假设每个字符的高度大约是字体大小的1.5倍

    estimated_text_height = max(int(height * 0.15), 25, int(width * 0.15))

    new_height = height + estimated_text_height
    new_image = Image.new('RGB', (width, new_height), color=(255, 255, 255))
    new_image.paste(image, (0, 0))

    # 添加字母A
    draw = ImageDraw.Draw(new_image)
    # 加载一个字体文件，这里使用默认字体
    # font = ImageFont.load_default()
    font = ImageFont.truetype("simfang.ttf", estimated_text_height - 5)
    # 由于之前的尝试都失败了，这里我们直接使用一个估计的文本高度

    # 计算文本的起始位置以使其居中

    text_position = ((width - 5) / 2, height + 5)
    draw.text(text_position, text, font=font, fill=(0, 0, 0))
    return new_image


def check_option_image(prompt):
    if '\nA.\nB.\nC.\nD.' in prompt:
        return True
    if '\nA.A\nB.B\nC.C\nD.D' in prompt:
        return True
    if '\nA、\nB、\nC、\nD、' in prompt:
        return True
    if '\nA、A\nB、B\nC、C\nD、D' in prompt:
        return True
    if '\nA．\nB．\nC．\nD．' in prompt:
        return True
    if '\nA．\nB．\nC．\nD．' in prompt:
        return True
    if '\nA．A\nB．B\nC．C\nD．D\n' in prompt:
        return True
    return False


def format_image(image_list, prompt):
    res_image_list = []
    add_option_on_image = len(image_list) == 4
    if not add_option_on_image:
        return image_list
    if not check_option_image(prompt):
        return image_list
    option_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    for i, image_path in enumerate(image_list):
        # image_path = image_path.replace('_new', '')
        save_image_path = os.path.join("raw_data/", image_path)

        open_image = Image.open(save_image_path)
        image_file_name = image_path.split("/")[-1]
        image_dir = "/".join(save_image_path.split("/")[:-1])
        image_file_name = ".".join(image_file_name.split(".")[:-1])
        new_save_image_path = os.path.join(image_dir, f"{image_file_name}_new.png")
        save_flag = False
        # if is_transparent(open_image):
        #     print(f'transparent image')
        #     open_image = convert_transparent_to_white_background(open_image)
        #     save_flag = True
        if add_option_on_image:
            open_image = add_option_to_image(open_image, option_dict[i])
            save_flag = True

        if save_flag:
            open_image.save(new_save_image_path)
            image_path = new_save_image_path.replace("raw_data/", "")

        res_image_list.append(image_path)
    return res_image_list


def check_illegal_image(image_list):
    for image_path in image_list:
        save_image_path = os.path.join("raw_data/", image_path)
        open_image = Image.open(save_image_path)
        if judge_illegal_image(open_image):
            print(f'过滤宽高不合法的图：width={open_image.width}, height={open_image.height}')
            return True
    return False


def check_empty_option(text):

    start_index = text.find('\nA、')

    text = text[start_index:].replace('\n你的答案是：', '').strip()
    if text[-1] == '。':
        text = text[:-1]
    if text == 'A、A\nB、B\nC、C\nD、D':
        return False
    if 'B、' not in text or 'C、' not in text or 'D、' not in text:
        print(f'缺少选项：{text}')
        return True

    text = text.split('\n')
    # if all(len(element) <= 3 for element in text):
    #     return False
    # if all(len(element) > 3 for element in text):
    #     return False

    content_list = []
    for x in text:
        x = x.replace('A、', '').replace('B、', '').replace('C、', '').replace('D、', '')
        if x in content_list:
            print(f'含有相同的选项内容：{text}')
            return True
        if len(x) == 0:
            print(f'过滤答案为空：{text}')

            return True
        content_list.append(x)

    return False


def clean_thread_data(data):
    res_data = []
    for line in tqdm(data, total=len(data), desc='clean'):
        prompt = line['conversations'][0]['value']
        answer = line['conversations'][1]['value']
        # 包含非法字母
        if not check_answer(answer):
            continue
        if line['keyword'] not in ['physics', 'chinese'] and '、' in line['conversations'][1]['value']:
            continue
        image_num = 0
        if 'image' in line:
            image_list = line['image']
            if not isinstance(line['image'], list):
                image_list = [line['image']]
            image_num = len(image_list)
            if check_illegal_image(image_list):
                continue

        prompt = reformat_prompt(prompt, image_num)
        prompt = clean_question(prompt)
        if prompt is None:
            continue
        if '\t' in prompt:
            continue
        if '\nA' in prompt and '\nB' in prompt and '\nC' in prompt and '\nD' in prompt:
            if '\nE' in prompt or '\nF' in prompt or '\nG' in prompt:
                continue
        if '\nA' not in prompt or '\nB' not in prompt or '\nC' not in prompt or '\nD' not in prompt:
            print(f'过滤非法题目:{prompt}')
            continue
        if prompt.count('选项是：') > 1:
            continue
        if check_empty_option(prompt):
            print(f'过滤不合法的答案：')
            print(prompt)

            continue
        # 有些就是一张图的badcase
        if image_num != 4 and '\nA、A\nB、B\nC、C\nD、D' in prompt:
            print(f'缺少选项图：{prompt}')
            continue

        line['conversations'][0]['value'] = prompt
        if 'image' in line:
            image_list = format_image(image_list, prompt)

            line['image'] = image_list
        # del_columns = ['common_info_in_file', 'common_info', 'html_path', 'question',
        #                'source_id', 'category', 'system', 'question_images', 'options']
        retain_columns = ['id', 'keyword', 'keyword_id', 'conversations', 'raw_answer',
                          'image', 'formular_data']
        keys = list(line.keys())
        for key in keys:
            if key not in retain_columns:
                line.pop(key)

        res_data.append(line)
    return res_data


def clean_data(data_path, save_path):
    data = load_json_file(data_path)
    # data = data[:30000]
    print(f'raw data num={len(data)}')
    thread_num = 32
    res_data = []
    every_thread_data = get_thread_data_with_image_num(data, thread_num, print_log=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        results = [
            executor.submit(clean_thread_data, thread_data) for thread_data in every_thread_data
        ]
        for future in concurrent.futures.as_completed(results):
            try:
                result = future.result()
                res_data.extend(result)
            except Exception as e:
                print(f"Thread generated an exception: {e}")

    print(f'clean data num={len(res_data)}')
    save_json_file(save_path, res_data)
    get_keyword_info(res_data)


if __name__ == '__main__':
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    print(f'data_path={data_path}\n'
          f'save_path={save_path}')
    clean_data(data_path, save_path)