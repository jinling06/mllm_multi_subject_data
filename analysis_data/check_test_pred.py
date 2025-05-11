"""
@Time: 2024/10/27 15:30
@Author: xujinlingbj
@File: check_test_pred.py
"""
import sys

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
        question = line['prompt']
        standard_answer = line['answer']
        pred_ans = line['pred_ans']
        current_score = line['current_score']
        html_parts.append(f"<p>**{i}**</p>")
        html_parts.append(f"<p>****</p>")
        html_parts.append(f"<p>question={question}</p>")
        html_parts.append(f"<p>answer={standard_answer}</p>")
        html_parts.append(f"<p>pred_answer={pred_ans}</p>")
        html_parts.append(f"<p>current_score={current_score}</p>")
        for image_path in picture:
            image_path = os.path.join('/mnt/data/xujinlingbj/LLaVA-MOSS2/raw_data', image_path)

            error_case_base64_img = image_file_to_base64(image_path)
            error_case_image_html = (
                f'<img src="data:image/png;base64,{error_case_base64_img}" alt="Sample Image" style="width:400px;">'
            )
            html_parts.append(error_case_image_html)

    html_parts = "\n".join(html_parts)
    return html_parts


def check_train_data(pred_data_path, test_data_path, out_html_path):
    pred_data = load_json_file(pred_data_path)
    pred_group_data = defaultdict(list)
    for line in pred_data:
        keyword = line['keyword']
        example = line['example']
        pred_group_data[keyword] = example
    test_data = load_json_file(test_data_path)
    group_data = defaultdict(list)
    for line in test_data:
        origin_index = line['origin_index']
        keyword = line['keyword']
        group_data[keyword].append([line, origin_index])

    key_2_html_info = {}
    for keyword, examples in group_data.items():

        examples = sorted(examples, key=lambda x: x[1])
        current_test_pred_res = pred_group_data[keyword]
        res_data = []
        for i in range(len(examples)):
            now_info = examples[i][0]
            answer = now_info['answer']
            answer = [x for x in answer]
            pred_ans = current_test_pred_res[i]['model_answer']
            assert current_test_pred_res[i]['index'] == now_info['origin_index']
            if pred_ans == answer:

                continue
            now_info['pred_ans'] = pred_ans
            now_info['current_score'] = current_test_pred_res[i]['current_score']
            res_data.append(now_info)
        print(keyword, f'data num={len(examples)}, pred false num={len(res_data)}, error ratio = {round(len(res_data) / len(examples), 4)*100}%')
        random.shuffle(res_data)
        html_parts = get_html_parts(res_data[:50])
        key_2_html_info[keyword] = {
            "title": keyword,
            "html": html_parts,
        }

    export_to_html(key_2_html_info, out_html_path)


if __name__ == '__main__':
    pred_data_path = '/mnt/data/xujinlingbj/LLAVA-NEXT-MLLM/output/submit-stage2-science-160k-1026-test-a-b-plain.json'
    test_data_path = '/mnt/data/xujinlingbj/LLaVA-MOSS2/raw_data/questions_a_b_for_pred.json'
    out_html_path = '/mnt/data/xujinlingbj/test_display_case/test_pred.html'
    check_train_data(pred_data_path, test_data_path, out_html_path)
    pass

