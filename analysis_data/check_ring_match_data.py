"""
@Time: 2024/10/9 11:25
@Author: xujinlingbj
@File: check_ring_match_data.py
"""
import base64
import io
import os.path
import random
import sys
from data_process.utils import *
from PIL import Image
from box_and_image_utils.draw_utils import export_to_html, image_file_to_base64
from tqdm import tqdm


def get_html_parts(data):
    print(f"data_num={len(data)}")

    html_parts = []
    print(f"html data num:{len(data)}")
    for i, line in tqdm(enumerate(data), desc="html"):
        picture = line['picture']
        question = line['question']
        standard_answer = line['standard_answer']
        html_parts.append(f"<p>**{i}**</p>")
        html_parts.append(f"<p>****</p>")
        html_parts.append(f"<p>question={question}</p>")
        html_parts.append(f"<p>standard_answer={standard_answer}</p>")
        for image_path in picture:
            image_path = os.path.join('/mnt/data/xujinlingbj/raw_data/ring_match_data/data', image_path)

            error_case_base64_img = image_file_to_base64(image_path)
            error_case_image_html = (
                f'<img src="data:image/png;base64,{error_case_base64_img}" alt="Sample Image" style="width:400px;">'
            )
            html_parts.append(error_case_image_html)

    html_parts = "\n".join(html_parts)
    return html_parts


def check_ring_match_data(data_path, out_html_path):
    data = load_json_file(data_path)
    group_data = defaultdict(list)
    key_2_html_info = {}
    for line in data:
        keyword = line['keyword']
        examples = line['example']
        random.shuffle(examples)
        html_parts = get_html_parts(examples[:10])
        key_2_html_info[keyword] = {
            "title": keyword,
            "html": html_parts,
        }

    export_to_html(key_2_html_info, out_html_path)


if __name__ == '__main__':
    data_path = '/mnt/data/xujinlingbj/raw_data/ring_match_data/data/questions.json'
    out_html_path = '/mnt/data/xujinlingbj/test_display_case/ring_match.html'
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    check_ring_match_data(data_path, out_html_path)
    pass