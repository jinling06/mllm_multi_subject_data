"""
@Time: 2024/10/27 14:45
@Author: xujinlingbj
@File: check_train_data.py
"""
from data_process.utils import *
from box_and_image_utils.draw_utils import export_to_html, image_file_to_base64


def get_html_parts(data):
    print(f"data_num={len(data)}")

    html_parts = []

    print(f"html data num:{len(data)}")
    for i, line in tqdm(enumerate(data), desc="html"):
        if 'image' in line:
            picture = line['image']
        else:
            picture = []
        if isinstance(picture, str):
            picture = [picture]
        idd = line['id']
        question = line['conversations'][0]['value']
        standard_answer = line['conversations'][1]['value']
        html_parts.append(f"<p>**{i}**</p>")
        html_parts.append(f"<p>**idd={idd}**</p>")
        html_parts.append(f"<p>question={question}</p>")
        html_parts.append(f"<p>answer={standard_answer}</p>")
        for image_path in picture:
            image_path = os.path.join('/mnt/data/xujinlingbj/raw_data', image_path)

            error_case_base64_img = image_file_to_base64(image_path)
            error_case_image_html = (
                f'<img src="data:image/png;base64,{error_case_base64_img}" alt="Sample Image" style="width:400px;">'
            )
            html_parts.append(error_case_image_html)

    html_parts = "\n".join(html_parts)
    return html_parts


def check_train_data(data_path, out_html_path):
    data = load_json_file(data_path)
    group_data = defaultdict(list)
    for line in data:
        if 'image' not in line:
            continue
        if len(line['image']) <2:
            continue
        keyword = line['keyword']
        group_data[keyword].append(line)

    key_2_html_info = {}
    for keyword, examples in group_data.items():
        # if keyword not in ['biology']:
        #     continue
        random.shuffle(examples)
        html_parts = get_html_parts(examples[:50])
        key_2_html_info[keyword] = {
            "title": keyword,
            "html": html_parts,
        }

    export_to_html(key_2_html_info, out_html_path)


if __name__ == '__main__':
    data_path = '/mnt/data/xujinlingbj/raw_data/llava_sft/stage2_train_1102_filter_clean.json'
    out_html_path = '/mnt/data/xujinlingbj/test_display_case/train.html'
    print(data_path)
    check_train_data(data_path, out_html_path)
    sample_data(
        data_path=data_path,
        output_path="./output/sample_train_random100.json",
    )
    pass

