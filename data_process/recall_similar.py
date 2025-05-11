"""
@Time: 2024/10/28 22:44
@Author: xujinlingbj
@File: recall_similar.py
"""
import json
import os
import sys
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from box_and_image_utils.draw_utils import image_file_to_base64, export_to_html
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import open_clip
from open_clip import create_model_and_transforms
from data_process.utils import *


def get_model(model_name, model_path, device):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    model.to(device)
    return model, preprocess, tokenizer


def get_image_embedding(model, preprocess, device, image_path):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(image,
                                        normalize=True
                                        )
    image_features = image_features.detach().cpu().numpy()
    return image_features


def build_database(model_name, model_path, device, data_path_list, output_dir, image_dir):
    data = []
    for data_path in data_path_list:
        data.extend(load_json_file(data_path))
    data = [x for x in data if 'image' in x]
    print(f'data num={len(data)}')
    model, preprocess, tokenizer = get_model(model_name, model_path, device)
    # 1152
    d = 1152
    # HNSW  Flat
    params = "L2Norm,Flat"
    index = faiss.index_factory(d, params, faiss.METRIC_INNER_PRODUCT)
    res_data = []
    for line in tqdm(data, total=len(data), desc='make'):
        images = line['image']
        if isinstance(images, str):
            images = [images]
        for image_path in images:
            image_path = os.path.join(image_dir, image_path)
            image_features = get_image_embedding(model, preprocess, device, image_path)
            if not index.is_trained:
                index.train(image_features)
            index.add(image_features)
            res_data.append({"image_path": image_path, 'infos': line})
    image_info_save_path = os.path.join(output_dir, 'image_index.json')
    print(f'res data num={len(res_data)}')
    save_json_file(image_info_save_path, res_data)
    faiss_save_path = os.path.join(output_dir, 'val_features.faiss')
    faiss.write_index(index, faiss_save_path)


def encode_image(image_path):
    error_case_base64_img = image_file_to_base64(image_path)
    error_case_image_html = (
        f'<img src="data:image/png;base64,{error_case_base64_img}" alt="Sample Image" style="width:400px;">'
    )
    return error_case_image_html


def search_similar(model_name, model_path, device, faiss_save_dir, query_data_path, out_html_path, query_image_dir):
    model, preprocess, tokenizer = get_model(model_name, model_path, device)
    image_info_save_path = os.path.join(faiss_save_dir, 'image_index.json')
    image_info_data = load_json_file(image_info_save_path)
    print(f'image_info_data={len(image_info_data)}')
    query_data = load_json_file(query_data_path)
    print(f'query_data num={len(query_data)}')
    faiss_save_path = os.path.join(faiss_save_dir, 'val_features.faiss')
    index = faiss.read_index(faiss_save_path)
    category_save_dict = defaultdict(int)
    html_parts = []
    cnt = 0
    for line in tqdm(query_data, total=len(query_data), desc='recall'):
        images = line['image']
        keyword = line['keyword']
        prompt = line['prompt']
        if category_save_dict[keyword] > 5:
            continue
        if isinstance(images, str):
            images = [images]
        for image_path in images:
            image_path = os.path.join(query_image_dir, image_path)
            image_features = get_image_embedding(model, preprocess, device, image_path)
            scores, idd = index.search(image_features, k=10)  # 实际的查询
            scores = scores[0].tolist()
            idd = idd[0].tolist()
            html_parts.append(f"<p>Number: {cnt}</p>")
            html_parts.append(f"<p>query: {prompt}</p>")
            html_parts.append(f'<p>score: {scores}</p>')
            html_parts.append(f'<p>image_path: {image_path}</p>')
            html_parts.append(encode_image(image_path))
            html_parts.append(f'<p>下面是召回的图</p>')
            for score, x in zip(scores, idd):
                recall_info = image_info_data[x]['infos']
                recall_image_path = image_info_data[x]['image_path']
                question = recall_info['conversations'][0]['value']

                # html_parts.append(f'<p>recall: {question}</p>')
                # html_parts.append(f'<p>recall_image_path: {recall_image_path}</p>')

                html_parts.append(encode_image(recall_image_path))

            cnt += 1
            category_save_dict[keyword] += 1
            break
    html_parts = "\n".join(html_parts)
    key_2_html_info = {}
    key_2_html_info["case"] = {
        "title": "case",
        "html": html_parts,
    }

    export_to_html(key_2_html_info, out_html_path)


if __name__ == '__main__':
    model_name = 'ViT-SO400M-14-SigLIP-384'
    model_path = 'model/ViT-SO400M-14-SigLIP-384/open_clip_pytorch_model.bin'
    device = "cuda:2"
    data_path_list = [
        'MLLM_data_2_v2/train_1027_with_formula_filter_with_image.json',
        'MLLM_data_2_v2/train_1027_arts.json'
    ]
    output_dir = './output/faiss_train_1028'
    image_dir = 'raw_data'
    os.makedirs(output_dir, exist_ok=True)
    # build_database(model_name, model_path, device, data_path_list, output_dir, image_dir)

    query_data_path = 'questions_a_b_for_pred.json'
    out_html_path = 'test_recall.html'
    query_image_dir = 'raw_data'
    search_similar(model_name, model_path, device, output_dir, query_data_path, out_html_path, query_image_dir)
