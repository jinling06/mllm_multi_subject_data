"""
@Time: 2024/10/28 17:53
@Author: xujinlingbj
@File: check_formula_data.py
"""
import random

from data_process.utils import *
from box_and_image_utils.draw_utils import export_to_html, image_file_to_base64


def get_html_parts(data):
    print(f"data_num={len(data)}")

    html_parts = []

    print(f"html data num:{len(data)}")
    for i, line in tqdm(enumerate(data), desc="html"):

        picture = [line['image_path']]
        if isinstance(picture, str):
            picture = [picture]

        model_pred = line['model_pred']
        clean_text = line['clean_text']
        html_parts.append(f"<p>**{i}**</p>")
        html_parts.append(f"<p>{picture}</p>")
        html_parts.append(f"<p>model_pred={model_pred}</p>")
        html_parts.append(f"<p>clean_text={clean_text}</p>")
        for image_path in picture:
            image_path = os.path.join('/mnt/data/xujinlingbj/raw_data', image_path)

            error_case_base64_img = image_file_to_base64(image_path)
            error_case_image_html = (
                f'<img src="data:image/png;base64,{error_case_base64_img}" alt="Sample Image" style="width:200px;">'
            )
            html_parts.append(error_case_image_html)

    html_parts = "\n".join(html_parts)
    return html_parts


def check_train_data(data_path, out_html_path):
    data = load_json_file(data_path)

    cnt = 0
    res_data = []
    for line in data:
        model_pred = line['model_pred']
        if check_illegal_formular_text(model_pred):
            cnt += 1
            print(model_pred)
            continue
        res_data.append(line)
    print(f'总共的图片数={len(data)}, 上面带箭头的图片={cnt}')
    random.shuffle(res_data)
    data = res_data[:100]
    key_2_html_info = {}

    html_parts = get_html_parts(data)
    key_2_html_info['case'] = {
        "title": 'case',
        "html": html_parts,
    }

    export_to_html(key_2_html_info, out_html_path)


if __name__ == '__main__':
    data_path = '/mnt/data/xujinlingbj/raw_data/llava_sft/math_data/MLLM_data_3_v5/image_path_to_formular_text_1031.json'
    out_html_path = '/mnt/data/xujinlingbj/test_display_case/formula.html'
    print(data_path)
    check_train_data(data_path, out_html_path)